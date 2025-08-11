# clinical_assistant_demo.py — Full Transcript Mode
import json, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
st.title("🩺 Clinical Intake – Chat Demo")
st.caption("Prototype only. Not medical advice.")

# Client (your key/secrets already configured)
client = OpenAI()

# ── State ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "demographics": {},
        "chief_complaint": None,
        "modules": {},
        "medications": [],
        "allergies": [],
        "red_flags": [],
        "free_text_notes": []
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if not st.session_state.messages:
    opening = "What brings you in today?"
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.asked_questions.append(opening)

# ── Prompt ──────────────────────────────────
SYSTEM = """
You are a clinical intake assistant for a primary care clinic.
We are sending you the FULL transcript each turn.

Output ONLY strict JSON with keys:
- next_question: string
- extracted_fields: object
- red_flags: string[]
- rationale: string

Rules (hard):
- ONE question only; short and specific. Prefer closed questions.
- NEVER repeat a question included in ASKED_QUESTIONS (rephrase only if clarifying a *new* detail).
- Detect red flags and put them ONLY in red_flags. NEVER hint at urgency/ER/red flags in next_question.
- If chief complaint unknown, ask: "What brings you in today?"

Normalization:
- Dates → ISO yyyy-mm-dd (approx ok: "~2025-08-01")
- Durations → {value:int, unit:"days|weeks|months|years"}
- Yes/No → true/false
- Meds → [{name, dose, route, frequency}]
- Allergies → [{substance, reaction}]
- Pain → {location, quality, severity_0_to_10, onset, duration, radiation, aggravating, relieving, associated}

Output ONLY JSON. No backticks, no prose outside JSON.
"""

ROUTING_HINT = """
Routing hint examples (do not repeat to user):
- chest/pressure/tightness → chest_pain module
- headache → headache module
- cough/fever/sore throat → URI module
"""

# ── Helpers ─────────────────────────────────
def merge_profile(profile: dict, extracted: dict) -> dict:
    if not extracted: return profile
    if extracted.get("chief_complaint") and not profile.get("chief_complaint"):
        profile["chief_complaint"] = extracted["chief_complaint"]
    if isinstance(extracted.get("demographics"), dict):
        profile["demographics"] = {**profile.get("demographics", {}), **extracted["demographics"]}
    if isinstance(extracted.get("medications"), list):
        seen = {(m.get("name"), m.get("dose"), m.get("route"), m.get("frequency")) for m in profile["medications"]}
        for m in extracted["medications"]:
            key = (m.get("name"), m.get("dose"), m.get("route"), m.get("frequency"))
            if key not in seen:
                profile["medications"].append(m); seen.add(key)
    if isinstance(extracted.get("allergies"), list):
        seen = {(a.get("substance"), a.get("reaction")) for a in profile["allergies"]}
        for a in extracted["allergies"]:
            key = (a.get("substance"), a.get("reaction"))
            if key not in seen:
                profile["allergies"].append(a); seen.add(key)
    if isinstance(extracted.get("modules"), dict):
        for mod, data in extracted["modules"].items():
            profile["modules"][mod] = {**profile["modules"].get(mod, {}), **data}
    if isinstance(extracted.get("free_text_notes"), list):
        profile["free_text_notes"].extend(extracted["free_text_notes"])
    return profile

def process_red_flags(new_flags):
    if not new_flags: return
    for f in new_flags:
        if f not in st.session_state.profile["red_flags"]:
            st.session_state.profile["red_flags"].append(f)

def full_messages(user_text: str):
    # Full transcript like ChatGPT (no truncation)
    msgs = [
        {"role":"system","content":SYSTEM},
        {"role":"system","content":f"PROFILE:{json.dumps(st.session_state.profile, ensure_ascii=False)}"},
        {"role":"system","content":f"ASKED_QUESTIONS:{json.dumps(st.session_state.asked_questions[-50:], ensure_ascii=False)}"},
        {"role":"system","content":ROUTING_HINT.strip()},
    ]
    msgs.extend(st.session_state.messages)  # ENTIRE transcript
    msgs.append({"role":"user","content":user_text})
    return msgs

def call_json(user_text: str):
    resp = client.chat.completions.create(
        model="gpt-4o",                   # ← higher-fidelity; swap to gpt-4o-mini if needed
        messages=full_messages(user_text),
        temperature=0.1,
        max_tokens=600,
        response_format={"type":"json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {
            "next_question": "Sorry, I didn’t catch that. Could you rephrase?",
            "extracted_fields": {},
            "red_flags": ["parse_error"],
            "rationale": "Non-JSON; safety fallback."
        }

def regenerate_advance(user_text: str, avoid_q: str):
    msgs = full_messages(user_text)
    msgs.append({"role":"user","content":f"[META] Do NOT repeat: {avoid_q}. Ask a different, advancing question."})
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        temperature=0.1,
        max_tokens=600,
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def clinician_summary(profile: dict) -> str:
    cc = profile.get("chief_complaint")
    demo = profile.get("demographics", {})
    mods = profile.get("modules", {})
    meds = profile.get("medications", [])
    algs = profile.get("allergies", [])
    rfs = profile.get("red_flags", [])
    lines = []
    bits = []
    if demo.get("age_years"): bits.append(f'{demo["age_years"]}y')
    if demo.get("sex"): bits.append(demo["sex"])
    if bits: lines.append("Patient: " + ", ".join(bits))
    if cc: lines.append(f"Chief complaint: {cc}")
    for mod, data in mods.items():
        show = []
        for k in ["onset","duration","location","quality","severity_0_to_10","radiation","aggravating","relieving","associated","red_flags_positive"]:
            if k in data: show.append(f"{k}={data[k]}")
        lines.append(f"{mod}: " + ("; ".join(show) if show else "(details captured)"))
    if meds:
        lines.append("Meds: " + "; ".join(
            f'{m.get("name","?")} {m.get("dose","")} {m.get("route","")} {m.get("frequency","")}'.strip()
            for m in meds
        ))
    if algs:
        lines.append("Allergies: " + "; ".join(
            f'{a.get("substance","?")}→{a.get("reaction","?")}' for a in algs
        ))
    if rfs: lines.append("⚠️ Red flags flagged: " + "; ".join(rfs))
    return "\n".join(lines) or "No structured data."

# ── UI ──────────────────────────────────────
with st.expander("Structured profile (debug)", expanded=False):
    st.json(st.session_state.profile, expanded=False)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_text = st.chat_input("Type your answer…")
if user_text:
    # 1) add user turn
    st.session_state.messages.append({"role":"user","content":user_text})

    # 2) model JSON
    data = call_json(user_text)

    # 3) merge + red flags (silent)
    st.session_state.profile = merge_profile(st.session_state.profile, data.get("extracted_fields", {}))
    process_red_flags(data.get("red_flags", []))

    # 4) next question
    next_q = (data.get("next_question") or "Please tell me more.").strip()

    # 5) de-dupe: if identical to last assistant question, regenerate once
    last_assistant = next((m["content"].strip() for m in reversed(st.session_state.messages) if m["role"]=="assistant"), "")
    if last_assistant and next_q.lower() == last_assistant.lower():
        try:
            data2 = regenerate_advance(user_text, last_assistant)
            st.session_state.profile = merge_profile(st.session_state.profile, data2.get("extracted_fields", {}))
            process_red_flags(data2.get("red_flags", []))
            next_q = (data2.get("next_question") or "Thanks—add one new detail.").strip()
        except Exception:
            next_q = "Thanks—add one new detail about this problem."

    # 6) append assistant question + track asked
    st.session_state.messages.append({"role":"assistant","content":next_q})
    st.session_state.asked_questions.append(next_q)

    st.rerun()

# Sidebar
with st.sidebar:
    st.header("Actions")
    if st.button("Create clinician summary"):
        st.code(clinician_summary(st.session_state.profile))
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.profile = {
            "demographics": {}, "chief_complaint": None, "modules": {},
            "medications": [], "allergies": [], "red_flags": [], "free_text_notes": []
        }
        st.session_state.asked_questions = []
        st.rerun()

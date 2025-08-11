# clinical_assistant_demo.py
# Streamlit clinical-intake chat with rolling summary + structured extraction
import os, json, datetime
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
st.title("🩺 Clinical Intake – Chat Demo")
st.caption("Please tell us your chief presenting complaint.")

# OpenAI client (key already loaded in your env/secrets per your setup)
client = OpenAI()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # full chat transcript (compact messages)
if "profile" not in st.session_state:
    st.session_state.profile = {
        "demographics": {},
        "chief_complaint": None,
        "modules": {},            # e.g., "chest_pain": {structured fields...}
        "medications": [],
        "allergies": [],
        "red_flags": [],
        "free_text_notes": []
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []  # track assistant questions to avoid repeats

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────
SYSTEM = """
You are a clinical intake assistant for a primary care clinic.
Goal: take a concise, safe medical history ONE question at a time.

Rules:
- Always return strict JSON (no prose) with keys:
  - next_question: string
  - extracted_fields: object (normalized, machine-readable)
  - red_flags: array of strings (empty if none)
  - rationale: short string (why you chose next question)
- Keep questions short and specific. Avoid chit-chat.
- Prefer closed questions where possible. Ask only ONE question per turn.
- If the patient’s answer is unsafe or suggests a red flag, include it in red_flags.
- If chief complaint is unknown, ask it first (“What brings you in today?”).

Do NOT repeat questions:
- Do not repeat any question listed in ASKED_QUESTIONS.
Red flags policy:
- Detect red flags and include them ONLY in the `red_flags` array.
- NEVER mention red flags, emergencies, urgency, ER, or disposition advice in `next_question`.
- Continue the interview with neutral, clinically appropriate questions.

Normalization guidance:
- Dates → ISO yyyy-mm-dd when possible (approximate allowed: “~2025-08-01”)
- Durations → {value:int, unit:"days|weeks|months|years"}
- Yes/No → true/false
- Medications → [{name, dose, route, frequency}]
- Allergies → [{substance, reaction}]
- Pain → {location, quality, severity_0_to_10:int, onset, duration, radiation, aggravating, relieving, associated}

Output ONLY JSON. No backticks, no explanations outside JSON.
"""

ROUTING_HINT = """
Routing hint examples (do not repeat to user):
- If chief complaint mentions "chest", "pressure", "tightness", consider chest_pain flow.
- If "headache" present, run headache flow.
- If "cough/fever/sore throat", run URI flow.
"""

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def merge_profile(profile: dict, extracted: dict) -> dict:
    if not extracted:
        return profile

    # Chief complaint
    if "chief_complaint" in extracted and extracted["chief_complaint"]:
        if not profile.get("chief_complaint"):
            profile["chief_complaint"] = extracted["chief_complaint"]

    # Demographics
    if "demographics" in extracted and isinstance(extracted["demographics"], dict):
        profile["demographics"] = {**profile.get("demographics", {}), **extracted["demographics"]}

    # Medications
    if "medications" in extracted and isinstance(extracted["medications"], list):
        seen = {(m.get("name"), m.get("dose"), m.get("route"), m.get("frequency")) for m in profile["medications"]}
        for m in extracted["medications"]:
            key = (m.get("name"), m.get("dose"), m.get("route"), m.get("frequency"))
            if key not in seen:
                profile["medications"].append(m)
                seen.add(key)

    # Allergies
    if "allergies" in extracted and isinstance(extracted["allergies"], list):
        seen = {(a.get("substance"), a.get("reaction")) for a in profile["allergies"]}
        for a in extracted["allergies"]:
            key = (a.get("substance"), a.get("reaction"))
            if key not in seen:
                profile["allergies"].append(a)
                seen.add(key)

    # Modules
    if "modules" in extracted and isinstance(extracted["modules"], dict):
        for mod, data in extracted["modules"].items():
            if mod not in profile["modules"]:
                profile["modules"][mod] = data
            else:
                profile["modules"][mod] = {**profile["modules"][mod], **data}

    # Free-text notes
    if "free_text_notes" in extracted and isinstance(extracted["free_text_notes"], list):
        profile["free_text_notes"].extend(extracted["free_text_notes"])

    return profile

def process_red_flags(new_flags):
    if not new_flags:
        return
    pf = st.session_state.profile
    for f in new_flags:
        if f not in pf["red_flags"]:
            pf["red_flags"].append(f)
    # patient UI remains unchanged (silent capture)

WINDOW_N = 5  # rolling context turns

def build_messages(user_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"PROFILE:{json.dumps(st.session_state.profile, ensure_ascii=False)}"},
        {"role": "system", "content": f"ASKED_QUESTIONS:{json.dumps(st.session_state.asked_questions[-20:], ensure_ascii=False)}"},
        {"role": "system", "content": ROUTING_HINT.strip()},
    ]
    tail = st.session_state.messages[-(2*WINDOW_N):]  # last N pairs
    msgs.extend(tail)
    msgs.append({"role": "user", "content": user_text})
    return msgs

def _model_call(messages):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def call_model(user_text: str):
    try:
        raw = _model_call(build_messages(user_text))
        return json.loads(raw)
    except Exception:
        return {
            "next_question": "Sorry, I didn’t catch that. Could you please rephrase?",
            "extracted_fields": {},
            "red_flags": ["parse_error"],
            "rationale": "Model returned non-JSON; safety fallback."
        }

def regenerate_with_avoid(user_text: str, avoid_q: str):
    # Nudge the model to choose a different question that advances the interview
    msgs = build_messages(user_text)
    meta = (f"[META] The question '{avoid_q}' was already asked and answered. "
            "Select a different next_question that advances the interview. Do not repeat.")
    msgs.append({"role": "user", "content": meta})
    raw = _model_call(msgs)
    return json.loads(raw)

def clinician_summary(profile: dict) -> str:
    cc = profile.get("chief_complaint")
    demo = profile.get("demographics", {})
    mods = profile.get("modules", {})
    meds = profile.get("medications", [])
    algs = profile.get("allergies", [])
    rfs = profile.get("red_flags", [])

    lines = []
    if demo:
        demo_bits = []
        if demo.get("age_years"): demo_bits.append(f'{demo["age_years"]}y')
        if demo.get("sex"): demo_bits.append(demo["sex"])
        if demo_bits:
            lines.append("Patient: " + ", ".join(demo_bits))
    if cc:
        lines.append(f"Chief complaint: {cc}")

    for mod, data in mods.items():
        snippet = []
        for k in ["onset", "duration", "location", "quality", "severity_0_to_10", "radiation",
                  "aggravating", "relieving", "associated", "red_flags_positive"]:
            if k in data:
                snippet.append(f"{k}={data[k]}")
        lines.append(f"{mod}: " + "; ".join(snippet) if snippet else f"{mod}: (details captured)")

    if meds:
        lines.append("Meds: " + "; ".join(
            f'{m.get("name","?")} {m.get("dose","")} {m.get("route","")} {m.get("frequency","")}'.strip()
            for m in meds
        ))
    if algs:
        lines.append("Allergies: " + "; ".join(
            f'{a.get("substance","?")}→{a.get("reaction","?")}' for a in algs
        ))
    if rfs:
        lines.append("⚠️ Red flags flagged: " + "; ".join(rfs))
    return "\n".join(lines) if lines else "No structured data."

# ─────────────────────────────────────────────
# First turn bootstrap (place BEFORE transcript rendering)
# ─────────────────────────────────────────────
if not st.session_state.messages:
    opening = "What brings you in today?"
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.asked_questions.append(opening.strip())

# ─────────────────────────────────────────────
# UI – transcript
# ─────────────────────────────────────────────
with st.expander("Structured profile (debug)", expanded=False):
    st.json(st.session_state.profile, expanded=False)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
user_text = st.chat_input("Type your answer…")
if user_text:
    # append user turn
    st.session_state.messages.append({"role": "user", "content": user_text})

    # call model for the next step (JSON)
    data = call_model(user_text)

    # merge extracted fields → rolling profile
    st.session_state.profile = merge_profile(st.session_state.profile, data.get("extracted_fields", {}))

    # process red flags silently
    process_red_flags(data.get("red_flags", []))

    # next question
    next_q = (data.get("next_question") or "Please tell me more.").strip()

    # de-dupe guard: compare to last assistant message
    last_assistant = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_assistant = m["content"].strip()
            break

    if last_assistant and next_q.lower() == last_assistant.lower():
        # regenerate with explicit avoid instruction
        try:
            data2 = regenerate_with_avoid(user_text, last_assistant)
            next_q = (data2.get("next_question") or "Okay, tell me more.").strip()
            # also merge any new fields/red flags that came with regeneration
            st.session_state.profile = merge_profile(
                st.session_state.profile, data2.get("extracted_fields", {})
            )
            process_red_flags(data2.get("red_flags", []))
        except Exception:
            # fallback if regen fails
            next_q = "Thanks. Could you add one new detail about this problem?"

    # record and show the new question
    st.session_state.messages.append({"role": "assistant", "content": next_q})
    st.session_state.asked_questions.append(next_q)

    st.rerun()

# ─────────────────────────────────────────────
# Sidebar – actions
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Actions")
    if st.button("Create clinician summary"):
        st.code(clinician_summary(st.session_state.profile))

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.profile = {
            "demographics": {},
            "chief_complaint": None,
            "modules": {},
            "medications": [],
            "allergies": [],
            "red_flags": [],
            "free_text_notes": []
        }
        st.session_state.asked_questions = []
        st.rerun()

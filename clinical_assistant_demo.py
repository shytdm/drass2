# clinical_assistant_demo.py — Simple "ChatGPT-style" full-history mode + AI summary
import json
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
st.title("🩺 Clinical Intake – Chat Demo")
st.caption("Prototype only. Not medical advice.")

client = OpenAI()  # your key is already configured

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "demographics": {},
        "chief_complaint": None,
        "modules": {},            # model may add blobs like {"chest_pain": {...}}
        "medications": [],
        "allergies": [],
        "red_flags": [],
        "free_text_notes": []
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None  # model-written clinician summary

# Seed opening question BEFORE rendering
if not st.session_state.messages:
    opening = "What brings you in today?"
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.asked_questions.append(opening)

# ─────────────────────────────────────────────
# System prompt (history taking & stop condition)
# ─────────────────────────────────────────────
SYSTEM = """
You are a clinical intake assistant for a primary care clinic.

Your task each session:
- Take a complete medical history for the patient's presenting complaint.
- Check all relevant red flags for that complaint.
- Gather essentials: past medical history, medications, allergies, family history, social history.
- Clarify anything unclear.
- Ask short, specific questions, ONE at a time.
- NEVER give medical advice. NEVER tell the patient about red flags or urgency. Red flags go ONLY in the JSON `red_flags` array for the doctor's note.

When you have enough information to form a proper differential diagnosis and a concise history, STOP asking questions and output:

{
  "next_question": "Thank you for answering my questions. Your history is being forwarded to your doctor!",
  "extracted_fields": { ... final structured history ... },
  "red_flags": [ ... ],
  "rationale": "History completed"
}

While interviewing, reply EACH TURN with ONLY strict JSON (no backticks, no extra prose) with keys:
- next_question: string  # the single next question OR the final thank-you line
- extracted_fields: object  # structured data parsed from the patient's last answer (mergeable)
- red_flags: string[]  # any detected red flags (for clinician only)
- rationale: string    # short reason for your next question/finish

Normalization:
- Dates → ISO yyyy-mm-dd when possible (approx allowed: "~2025-08-01")
- Durations → {value:int, unit:"days|weeks|months|years"}
- Yes/No → true/false
- Medications → [{name, dose, route, frequency}]
- Allergies → [{substance, reaction}]
- Pain → {location, quality, severity_0_to_10, onset, duration, radiation, aggravating, relieving, associated}

De-duplication:
- Do NOT repeat the last assistant question verbatim. If clarifying, rephrase and advance.

Output ONLY the JSON object. Nothing else.
"""

ROUTING_HINT = """
Routing hint examples (do not repeat to user):
- chest/pressure/tightness → capture chest pain details + red flags
- headache → headache details + red flags
- cough/fever/sore throat → URI details + red flags
"""

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def merge_profile(profile: dict, extracted: dict) -> dict:
    if not extracted:
        return profile
    # Chief complaint
    if extracted.get("chief_complaint") and not profile.get("chief_complaint"):
        profile["chief_complaint"] = extracted["chief_complaint"]
    # Demographics
    if isinstance(extracted.get("demographics"), dict):
        profile["demographics"] = {**profile.get("demographics", {}), **extracted["demographics"]}
    # Medications
    if isinstance(extracted.get("medications"), list):
        seen = {(m.get("name"), m.get("dose"), m.get("route"), m.get("frequency")) for m in profile["medications"]}
        for m in extracted["medications"]:
            key = (m.get("name"), m.get("dose"), m.get("route"), m.get("frequency"))
            if key not in seen:
                profile["medications"].append(m); seen.add(key)
    # Allergies
    if isinstance(extracted.get("allergies"), list):
        seen = {(a.get("substance"), a.get("reaction")) for a in profile["allergies"]}
        for a in extracted["allergies"]:
            key = (a.get("substance"), a.get("reaction"))
            if key not in seen:
                profile["allergies"].append(a); seen.add(key)
    # Modules (symptom-specific blobs)
    if isinstance(extracted.get("modules"), dict):
        for mod, data in extracted["modules"].items():
            profile["modules"][mod] = {**profile["modules"].get(mod, {}), **data}
    # Free-text notes
    if isinstance(extracted.get("free_text_notes"), list):
        profile["free_text_notes"].extend(extracted["free_text_notes"])
    return profile

def process_red_flags(new_flags):
    if not new_flags: return
    for f in new_flags:
        if f not in st.session_state.profile["red_flags"]:
            st.session_state.profile["red_flags"].append(f)

def full_messages(user_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"PROFILE:{json.dumps(st.session_state.profile, ensure_ascii=False)}"},
        {"role": "system", "content": f"ASKED_QUESTIONS:{json.dumps(st.session_state.asked_questions[-50:], ensure_ascii=False)}"},
        {"role": "system", "content": ROUTING_HINT.strip()},
    ]
    msgs.extend(st.session_state.messages)  # ENTIRE transcript
    msgs.append({"role": "user", "content": user_text})
    return msgs

def call_json(user_text: str):
    resp = client.chat.completions.create(
        model="gpt-4o",  # use 4o for fidelity; swap to -mini if needed
        messages=full_messages(user_text),
        temperature=0.1,
        max_tokens=700,
        response_format={"type": "json_object"},
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
        max_tokens=700,
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ---- Clinician Summary via ChatGPT ------------------------------------------
SUMMARY_TASK = """You are assisting a clinician.

TASK:
- Write a concise clinical summary (max 150 words)
- Highlight red flags
- Give 2–4 differential diagnoses
- Recommend next steps and first-line treatment
- Use bullet points and medical clarity. No disclaimers.

INPUTS:
- Structured patient profile (JSON) with demographics, chief complaint, modules (symptom details), medications, allergies, red flags, and notes.
- Conversation transcript (user/assistant turns). Use it only to clarify context; prefer structured data as source of truth.

OUTPUT:
Return plain markdown bullets — no preamble, no code fences.
"""

def generate_clinician_summary_via_model(profile: dict, transcript: list[str]) -> str:
    # Build a single prompt with profile + a compact transcript
    # Keep transcript short-ish to control tokens
    compact_transcript = []
    for m in transcript[-60:]:  # last 60 turns max
        role = m.get("role", "")
        text = m.get("content", "")
        if role in ("user", "assistant"):
            compact_transcript.append(f"{role.upper()}: {text}")
    transcript_text = "\n".join(compact_transcript)

    messages = [
        {"role": "system", "content": SUMMARY_TASK},
        {"role": "user", "content":
            "STRUCTURED_PROFILE_JSON:\n" + json.dumps(profile, ensure_ascii=False) +
            "\n\nTRANSCRIPT:\n" + transcript_text +
            "\n\nWrite the summary now."}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

def clinician_summary_from_profile_only(profile: dict) -> str:
    # Deterministic local fallback (no API) if you ever want it
    cc = profile.get("chief_complaint")
    demo = profile.get("demographics", {})
    mods = profile.get("modules", {})
    meds = profile.get("medications", [])
    algs = profile.get("allergies", [])
    rfs = profile.get("red_flags", [])
    lines = []
    lines.append("• Summary:")
    bits = []
    if demo.get("age_years"): bits.append(f'{demo["age_years"]}y')
    if demo.get("sex"): bits.append(demo["sex"])
    if cc: bits.append(f"CC: {cc}")
    if bits: lines.append("  - " + ", ".join(bits))
    for mod, data in mods.items():
        show = []
        for k in ["onset","duration","location","quality","severity_0_to_10","radiation","aggravating","relieving","associated"]:
            if k in data: show.append(f"{k}={data[k]}")
        if show: lines.append(f"  - {mod}: " + "; ".join(show))
    if meds: lines.append("• Meds: " + "; ".join(f'{m.get("name","?")} {m.get("dose","")} {m.get("route","")} {m.get("frequency","")}'.strip() for m in meds))
    if algs: lines.append("• Allergies: " + "; ".join(f'{a.get("substance","?")}→{a.get("reaction","?")}' for a in algs))
    if rfs: lines.append("• ⚠️ Red flags: " + "; ".join(rfs))
    lines.append("• Differentials: —")
    lines.append("• Next steps: —")
    lines.append("• First-line: —")
    return "\n".join(lines)

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
    # 1) append user turn
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2) get JSON step
    data = call_json(user_text)

    # 3) merge structured data and capture red flags silently
    st.session_state.profile = merge_profile(st.session_state.profile, data.get("extracted_fields", {}))
    process_red_flags(data.get("red_flags", []))

    # 4) next question or completion
    next_q = (data.get("next_question") or "Please tell me more.").strip()

    # 5) de-dupe (avoid repeating last assistant question)
    last_assistant = next((m["content"].strip() for m in reversed(st.session_state.messages) if m["role"] == "assistant"), "")
    if last_assistant and next_q.lower() == last_assistant.lower():
        try:
            data2 = regenerate_advance(user_text, last_assistant)
            st.session_state.profile = merge_profile(st.session_state.profile, data2.get("extracted_fields", {}))
            process_red_flags(data2.get("red_flags", []))
            next_q = (data2.get("next_question") or "Thanks—please add one new detail.").strip()
        except Exception:
            next_q = "Thanks—please add one new detail."

    # 6) if completion message, show it, generate clinician summary with GPT, and stop
    if next_q.lower().startswith("thank you for answering my questions"):
        st.session_state.messages.append({"role": "assistant", "content": next_q})
        st.session_state.asked_questions.append(next_q)

        # Generate clinician summary via model (primary) with profile + transcript
        try:
            st.session_state.final_summary = generate_clinician_summary_via_model(
                st.session_state.profile,
                st.session_state.messages
            )
        except Exception:
            # fallback to local summary if API hiccups
            st.session_state.final_summary = clinician_summary_from_profile_only(st.session_state.profile)

        # Render summary
        with st.sidebar:
            st.header("Clinician summary (AI-generated)")
            st.markdown(st.session_state.final_summary)

        st.stop()

    # 7) otherwise continue the interview
    st.session_state.messages.append({"role": "assistant", "content": next_q})
    st.session_state.asked_questions.append(next_q)
    st.rerun()

# ─────────────────────────────────────────────
# Sidebar – actions
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Actions")
    if st.button("Generate clinician summary now"):
        try:
            st.session_state.final_summary = generate_clinician_summary_via_model(
                st.session_state.profile,
                st.session_state.messages
            )
        except Exception:
            st.session_state.final_summary = clinician_summary_from_profile_only(st.session_state.profile)
    if st.session_state.final_summary:
        st.markdown(st.session_state.final_summary)

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
        st.session_state.final_summary = None
        st.rerun()

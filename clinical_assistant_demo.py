# clinical_assistant_demo.py — Demographics form + ChatGPT-style intake + deterministic stop + 30Q cap + AI summary
import json
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
st.title("🩺 Clinical Intake – Chat Demo")
st.caption("Prototype only. Not medical advice.")

client = OpenAI()  # key already configured in your env/secrets

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "demographics": {},           # {name, age_years, sex, ...}
        "chief_complaint": None,
        "modules": {},                # symptom-specific blobs (model-defined)
        "medications": [],            # [{name,dose,route,frequency}]
        "allergies": [],              # [{substance,reaction}]
        "past_medical_history": None, # str or dict
        "family_history": None,       # str or dict
        "social_history": None,       # str or dict
        "red_flags": [],              # collected silently
        "red_flags_checked": False,   # set true once screened
        "free_text_notes": []
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None
if "q_count" not in st.session_state:
    st.session_state.q_count = 0  # assistant questions asked (excl. final thank-you)

MAX_QUESTIONS = 30

# ─────────────────────────────────────────────
# Static Demographics Form (above chat)
# ─────────────────────────────────────────────
with st.form("demographics_form", clear_on_submit=False):
    st.subheader("Patient Demographics")
    name = st.text_input("Full Name", value=st.session_state.profile["demographics"].get("name", ""))

    age_val = st.session_state.profile["demographics"].get("age_years")
    age = st.number_input(
        "Age (years)", min_value=0, max_value=120, step=1,
        value=int(age_val) if isinstance(age_val, (int, float)) and age_val is not None else 0
    )

    sex_options = ["", "Male", "Female", "Intersex", "Prefer not to say"]
    stored_sex = st.session_state.profile["demographics"].get("sex") or ""
    try:
        sex_index = sex_options.index(stored_sex)
    except ValueError:
        sex_index = 0
    sex = st.selectbox("Sex at Birth", sex_options, index=sex_index)

    submit_demo = st.form_submit_button("Save")

    if submit_demo:
        st.session_state.profile["demographics"]["name"] = name.strip() or None
        st.session_state.profile["demographics"]["age_years"] = age if age > 0 else None
        st.session_state.profile["demographics"]["sex"] = sex or None
        st.success("Demographics saved.")

# ─────────────────────────────────────────────
# Seed opening question BEFORE transcript rendering
# ─────────────────────────────────────────────
if not st.session_state.messages:
    opening = "What brings you in today?"
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.asked_questions.append(opening)
    st.session_state.q_count = 1

# ─────────────────────────────────────────────
# Deterministic stop rules
# ─────────────────────────────────────────────
REQUIRED_SECTIONS = [
    "chief_complaint",
    "demographics",
    "past_medical_history",
    "medications",
    "allergies",
    "family_history",
    "social_history",
    "red_flags_checked",
]

def history_complete(profile: dict) -> bool:
    if not profile.get("chief_complaint"):
        return False
    demo = profile.get("demographics", {})
    if not (demo.get("age_years") or demo.get("sex")):
        return False
    if not profile.get("past_medical_history"):
        return False
    if not profile.get("family_history"):
        return False
    if not profile.get("social_history"):
        return False
    if not bool(profile.get("red_flags_checked")):
        return False
    return True

# ─────────────────────────────────────────────
# System prompt (history taking & stop signal for the MODEL)
# ─────────────────────────────────────────────
SYSTEM = """
You are a clinical intake assistant for a primary care clinic.

Your task each session:
- Take a complete medical history for the patient's presenting complaint.
- Check all relevant red flags for that complaint.
- Gather essentials: past medical history, medications, allergies, family history, social history.
- Clarify anything unclear.
- Ask short, specific questions, ONE at a time.
- NEVER give medical advice. NEVER tell the patient about red flags or urgency. Red flags go ONLY in the JSON red_flags array for the doctor's note.

IMPORTANT:
- Set red_flags_checked to true in extracted_fields once you have screened red flags relevant to the complaint.
- Always parse any new info into extracted_fields so the app can track completion.
- Do NOT repeat the last assistant question verbatim. If clarifying, rephrase and advance.

When you have enough information to form a proper differential diagnosis and a concise history, STOP asking questions and output:

{
  "next_question": "Thank you for answering my questions. Your history is being forwarded to your doctor!",
  "extracted_fields": { ... final structured history ... },
  "red_flags": [ ... ],
  "rationale": "History completed"
}

Each turn, reply with ONLY strict JSON (no backticks, no extra prose) with keys:
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

    if extracted.get("past_medical_history"):
        profile["past_medical_history"] = extracted["past_medical_history"]
    if extracted.get("family_history"):
        profile["family_history"] = extracted["family_history"]
    if extracted.get("social_history"):
        profile["social_history"] = extracted["social_history"]

    if isinstance(extracted.get("modules"), dict):
        for mod, data in extracted.items():
            if mod == "modules" and isinstance(data, dict):
                for submod, subdata in data.items():
                    profile["modules"][submod] = {**profile["modules"].get(submod, {}), **subdata}

    if isinstance(extracted.get("free_text_notes"), list):
        profile["free_text_notes"].extend(extracted["free_text_notes"])

    if "red_flags_checked" in extracted:
        profile["red_flags_checked"] = bool(extracted["red_flags_checked"])

    return profile

def process_red_flags(new_flags):
    if not new_flags:
        return
    for f in new_flags:
        if f not in st.session_state.profile["red_flags"]:
            st.session_state.profile["red_flags"].append(f)

def full_messages():
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"PROFILE:{json.dumps(st.session_state.profile, ensure_ascii=False)}"},
        {"role": "system", "content": f"ASKED_QUESTIONS:{json.dumps(st.session_state.asked_questions[-50:], ensure_ascii=False)}"},
        {"role": "system", "content": ROUTING_HINT.strip()},
    ]
    msgs.extend(st.session_state.messages)  # ENTIRE transcript (already includes the latest user turn)
    return msgs

def call_json():
    resp = client.chat.completions.create(
        model="gpt-4o",  # for fidelity; swap to -mini later if needed
        messages=full_messages(),
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

def regenerate_advance(avoid_q: str):
    msgs = full_messages()
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
- Structured patient profile (JSON) with demographics, chief complaint, modules (symptom details), medications, allergies, past medical history, family history, social history, red flags, and notes.
- Conversation transcript (user/assistant turns). Use it only to clarify context; prefer structured data as source of truth.

OUTPUT:
Return plain markdown bullets — no preamble, no code fences.
"""

def generate_clinician_summary_via_model(profile: dict, transcript: list[dict]) -> str:
    compact_transcript = []
    for m in transcript[-60:]:  # last 60 turns max
        role = m.get("role", "")
        text = m.get("content", "")
        if role in ("user", "assistant"):
            compact_transcript.append(f"{str(role).upper()}: {text}")
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

def finish_and_summarize():
    """Emit final message, generate summary, and stop the app run."""
    final_message = "Thank you for answering my questions. Your history is being forwarded to your doctor!"
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    st.session_state.asked_questions.append(final_message)
    try:
        st.session_state.final_summary = generate_clinician_summary_via_model(
            st.session_state.profile,
            st.session_state.messages
        )
    except Exception as e:
        st.session_state.final_summary = None
        with st.sidebar:
            st.error("Summary generation failed.")
            st.exception(e)
    with st.sidebar:
        st.header("Clinician summary (AI-generated)")
        if st.session_state.final_summary:
            st.markdown(st.session_state.final_summary)
        else:
            st.write("Summary unavailable. Try again from the sidebar.")
    st.stop()

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

    # 2) get JSON step (no duplicate user turn injected)
    data = call_json()

    # 3) merge structured data and capture red flags silently
    st.session_state.profile = merge_profile(st.session_state.profile, data.get("extracted_fields", {}))
    process_red_flags(data.get("red_flags", []))

    # 4) deterministic stop checks BEFORE asking the next question
    if history_complete(st.session_state.profile) or st.session_state.q_count >= MAX_QUESTIONS:
        finish_and_summarize()

    # 5) next question or completion (model-driven)
    next_q = (data.get("next_question") or "Please tell me more.").strip()

    # 6) de-dupe (avoid repeating last assistant question)
    last_assistant = next(
        (m["content"].strip() for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        ""
    )
    if last_assistant and next_q.lower() == last_assistant.lower():
        try:
            data2 = regenerate_advance(last_assistant)
            st.session_state.profile = merge_profile(st.session_state.profile, data2.get("extracted_fields", {}))
            process_red_flags(data2.get("red_flags", []))
            next_q = (data2.get("next_question") or "Thanks—please add one new detail.").strip()
        except Exception:
            next_q = "Thanks—please add one new detail."

    # 7) if model announced completion, finish
    if next_q.lower().startswith("thank you for answering my questions"):
        finish_and_summarize()

    # 8) otherwise continue the interview
    st.session_state.messages.append({"role": "assistant", "content": next_q})
    st.session_state.asked_questions.append(next_q)
    st.session_state.q_count += 1
    if st.session_state.q_count >= MAX_QUESTIONS:
        finish_and_summarize()

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
            st.markdown(st.session_state.final_summary)
        except Exception as e:
            st.warning("Summary generation failed. Please try again.")
            st.exception(e)
    st.markdown(f"**Questions asked:** {st.session_state.q_count} / {MAX_QUESTIONS}")

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.profile = {
            "demographics": {},
            "chief_complaint": None,
            "modules": {},
            "medications": [],
            "allergies": [],
            "past_medical_history": None,
            "family_history": None,
            "social_history": None,
            "red_flags": [],
            "red_flags_checked": False,
            "free_text_notes": []
        }
        st.session_state.asked_questions = []
        st.session_state.final_summary = None
        st.session_state.q_count = 0
        st.rerun()

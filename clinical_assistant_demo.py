# clinical_assistant_demo_v2.py — Demographics form + ChatGPT-style intake + deterministic finish + 30Q cap + AI summary
# (Implements: finish flag, missing_fields, info_gain pressure, additive merges, de-dupe, red-flag checklist, progress panel)

import json
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
st.title("🩺 Clinical Intake – Chat Demo (v2)")
st.caption("Prototype only. Not medical advice.")

client = OpenAI()  # key already configured in your env/secrets

# ─────────────────────────────────────────────
# Constants & Checklists
# ─────────────────────────────────────────────
MAX_QUESTIONS = 30

# Minimal red-flag checklist (extend as needed)
RED_FLAG_CHECKLIST = {
    "chest_pain": [
        "exertional", "radiation", "dyspnea", "syncope", "diaphoresis", "risk_factors"
    ],
    "headache": [
        "thunderclap", "neurologic_deficit", "neck_stiffness", "fever", "immunosuppression"
    ],
    "abdominal_pain": [
        "rigidity", "rebound", "fever", "GI_bleed", "pregnancy", "jaundice"
    ],
    "sore_throat": [
        "airway_compromise", "drooling", "trismus", "peritonsillar_abscess_signs", "dehydration"
    ],
    "shortness_of_breath": [
        "hypoxia", "tachypnea", "chest_pain", "hemoptysis", "unilateral_leg_swelling"
    ]
}

ROUTING_HINT = """
Routing hint examples (do not repeat to user):
- chest/pressure/tightness → capture chest pain details + red flags
- headache → headache details + red flags
- cough/fever/sore throat → URI details + red flags
- abdominal pain → pain module + GI red flags
- shortness of breath/dyspnea → dyspnea module + cardiorespiratory red flags
"""

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "demographics": {},            # {name, age_years, sex, ...}
        "chief_complaint": None,
        "modules": {},                 # symptom-specific blobs (model-defined)
        "medications": [],             # [{name,dose,route,frequency}]
        "allergies": [],               # [{substance,reaction}]
        "past_medical_history": {},    # dict; additive
        "family_history": {},          # dict; additive
        "social_history": {},          # dict; additive
        "red_flags": [],               # collected silently
        "red_flags_checked": False,    # set true once screened
        "free_text_notes": []
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None
if "q_count" not in st.session_state:
    st.session_state.q_count = 0  # assistant questions asked (excl. final thank-you)
if "missing_fields_latest" not in st.session_state:
    st.session_state.missing_fields_latest = []

# ─────────────────────────────────────────────
# Static Demographics Form (above chat)
# ─────────────────────────────────────────────
with st.form("demographics_form", clear_on_submit=False):
    st.subheader("Patient Demographics")
    name = st.text_input("Full Name", value=st.session_state.profile["demographics"].get("name", ""))
    age = st.number_input(
        "Age (years)", min_value=0, max_value=120, step=1,
        value=int(st.session_state.profile["demographics"].get("age_years") or 0)
    )
    sex = st.selectbox(
        "Sex at Birth",
        ["", "Male", "Female", "Intersex", "Prefer not to say"],
        index=["", "Male", "Female", "Intersex", "Prefer not to say"].index(
            st.session_state.profile["demographics"].get("sex", "")
        )
    )
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
# System prompt (history taking with finish gating)
# ─────────────────────────────────────────────
SYSTEM = f"""
You are a clinical intake assistant for a primary care clinic.

GOAL
- Get a complete, decision-useful history with the FEWEST questions.
- Ask ONE short, specific question per turn.
- NEVER give medical advice. NEVER reveal red flags to patient.
- Prefer structured facts over prose. Do not repeat already-answered or explicitly-unknown items.

SLOT MODEL (fill these if applicable)
Required core slots:
- chief_complaint (string)
- demographics.age_years (int), demographics.sex (string)
- past_medical_history (dict)
- medications (list of {{name,dose,route,frequency}})
- allergies (list of {{substance,reaction}})
- family_history (dict)
- social_history (dict: tobacco/alcohol/drugs/occupation/living)
- red_flags_checked (boolean)
Symptom modules (only if relevant to CC), e.g.:
- pain: {{location, quality, severity_0_to_10, onset, duration:{{value,unit}}, radiation, aggravating, relieving, associated}}

RED FLAG POLICY
- For common complaints (chest pain, SOB, headache, abdominal pain, fever/URI), you MUST complete a standard checklist before setting red_flags_checked=true.
- Use this checklist map (JSON): {json.dumps(RED_FLAG_CHECKLIST, ensure_ascii=False)}
- Put red flag observations ONLY in `red_flags` array.

STOP CRITERIA (strict)
- Compute `missing_fields`: slots still empty that would change DDx or immediate next steps.
- Estimate `info_gain_estimate` in [0,1] for the NEXT question.
- If `missing_fields` is empty OR `info_gain_estimate < 0.15`, set `finish=true`.
- Otherwise `finish=false` and ask exactly ONE new question targeting the highest-yield missing slot(s).

DE-DUPE
- Use provided ASKED_QUESTIONS to avoid repeats. If a slot is answered or marked unknown, never ask it again. Track that in `missing_fields_ignored`.

BUDGET
- You have a remaining question budget. Ask only if the answer is likely to change DDx or immediate plan.

OUTPUT FORMAT — STRICT JSON ONLY (no prose):
{{
  "next_question": "string (or final thank-you line if finish=true)",
  "finish": true|false,
  "extracted_fields": {{...mergeable structured data...}},
  "red_flags": ["..."],
  "missing_fields": ["slot.path", "..."],
  "missing_fields_ignored": ["slot.path", "..."],
  "target_slots": ["slot.path", "..."],
  "info_gain_estimate": 0.0,
  "rationale": "≤15 words about why this question or finish"
}}

Normalization:
- Dates → ISO yyyy-mm-dd when possible (approx allowed: "~2025-08-01")
- Durations → {{value:int, unit:"days|weeks|months|years"}}
- Yes/No → true/false
- Medications → [{{name, dose, route, frequency}}]
- Allergies → [{{substance, reaction}}]
- Pain → {{location, quality, severity_0_to_10, onset, duration, radiation, aggravating, relieving, associated}}

Output ONLY the JSON object. Nothing else.
"""

# ─────────────────────────────────────────────
# Helpers — merging & red flags
# ─────────────────────────────────────────────

def _merge_dict(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v

def merge_profile(profile: dict, extracted: dict) -> dict:
    if not extracted:
        return profile

    # chief complaint once
    if extracted.get("chief_complaint") and not profile.get("chief_complaint"):
        profile["chief_complaint"] = extracted["chief_complaint"]

    # demographics
    if isinstance(extracted.get("demographics"), dict):
        _merge_dict(profile.setdefault("demographics", {}), extracted["demographics"])

    # medications (set-like)
    if isinstance(extracted.get("medications"), list):
        seen = {(m.get("name"), m.get("dose"), m.get("route"), m.get("frequency")) for m in profile["medications"]}
        for m in extracted["medications"]:
            key = (m.get("name"), m.get("dose"), m.get("route"), m.get("frequency"))
            if key not in seen:
                profile["medications"].append(m)
                seen.add(key)

    # allergies (set-like)
    if isinstance(extracted.get("allergies"), list):
        seen = {(a.get("substance"), a.get("reaction")) for a in profile["allergies"]}
        for a in extracted["allergies"]:
            key = (a.get("substance"), a.get("reaction"))
            if key not in seen:
                profile["allergies"].append(a)
                seen.add(key)

    # dict merges for histories
    for k in ["past_medical_history", "family_history", "social_history"]:
        if isinstance(extracted.get(k), dict):
            _merge_dict(profile.setdefault(k, {}), extracted[k])
        elif extracted.get(k):  # string or other
            profile[k] = extracted[k]

    # modules
    if isinstance(extracted.get("modules"), dict):
        for mod, data in extracted["modules"].items():
            _merge_dict(profile.setdefault("modules", {}).setdefault(mod, {}), data)

    # notes
    if isinstance(extracted.get("free_text_notes"), list):
        profile["free_text_notes"].extend(extracted["free_text_notes"])

    # red-flag gate
    if "red_flags_checked" in extracted:
        profile["red_flags_checked"] = bool(extracted["red_flags_checked"])

    return profile


def process_red_flags(new_flags):
    if not new_flags:
        return
    for f in new_flags:
        if f not in st.session_state.profile["red_flags"]:
            st.session_state.profile["red_flags"].append(f)


def full_messages(user_text: str):
    # Question budget hint
    remaining_budget = max(0, MAX_QUESTIONS - st.session_state.q_count)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"PROFILE:{json.dumps(st.session_state.profile, ensure_ascii=False)}"},
        {"role": "system", "content": f"ASKED_QUESTIONS:{json.dumps(st.session_state.asked_questions[-50:], ensure_ascii=False)}"},
        {"role": "system", "content": f"QUESTION_BUDGET:{remaining_budget}"},
        {"role": "system", "content": ROUTING_HINT.strip()},
    ]
    msgs.extend(st.session_state.messages)  # ENTIRE transcript
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _response_format_json_object():
    # Portable default; if your runtime supports json_schema, you can swap it in below.
    return {"type": "json_object"}


def _response_format_json_schema():
    # If supported by your SDK/runtime, this constrains output better.
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intake_step",
            "schema": {
                "type": "object",
                "required": [
                    "next_question", "finish", "extracted_fields", "red_flags",
                    "missing_fields", "target_slots", "info_gain_estimate", "rationale"
                ],
                "properties": {
                    "next_question": {"type": "string"},
                    "finish": {"type": "boolean"},
                    "extracted_fields": {"type": "object"},
                    "red_flags": {"type": "array", "items": {"type": "string"}},
                    "missing_fields": {"type": "array", "items": {"type": "string"}},
                    "missing_fields_ignored": {"type": "array", "items": {"type": "string"}},
                    "target_slots": {"type": "array", "items": {"type": "string"}},
                    "info_gain_estimate": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 120}
                },
                "additionalProperties": True
            }
        }
    }


USE_JSON_SCHEMA = False  # set True if your SDK supports response_format json_schema


def call_json(user_text: str):
    rf = _response_format_json_schema() if USE_JSON_SCHEMA else _response_format_json_object()
    resp = client.chat.completions.create(
        model="gpt-4o",  # for fidelity; swap to -mini later if needed
        messages=full_messages(user_text),
        temperature=0.1,
        max_tokens=700,
        response_format=rf,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        return {
            "next_question": "Sorry, I didn’t catch that. Could you rephrase?",
            "finish": False,
            "extracted_fields": {},
            "red_flags": ["parse_error"],
            "missing_fields": [],
            "missing_fields_ignored": [],
            "target_slots": [],
            "info_gain_estimate": 0.0,
            "rationale": "Non-JSON; safety fallback."
        }

    # Soft validation & defaults
    data.setdefault("finish", False)
    data.setdefault("extracted_fields", {})
    data.setdefault("red_flags", [])
    data.setdefault("missing_fields", [])
    data.setdefault("missing_fields_ignored", [])
    data.setdefault("target_slots", [])
    data.setdefault("info_gain_estimate", 0.0)
    data.setdefault("rationale", "")
    data.setdefault("next_question", "Please tell me more.")
    return data


def regenerate_advance(user_text: str, avoid_q: str):
    msgs = full_messages(user_text)
    msgs.append({"role": "user", "content": f"[META] Do NOT repeat: {avoid_q}. Ask a different, advancing question."})
    rf = _response_format_json_schema() if USE_JSON_SCHEMA else _response_format_json_object()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        temperature=0.1,
        max_tokens=700,
        response_format=rf,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {
            "next_question": "Thanks — please add one new detail.",
            "finish": False,
            "extracted_fields": {},
            "red_flags": [],
            "missing_fields": [],
            "missing_fields_ignored": [],
            "target_slots": [],
            "info_gain_estimate": 0.0,
            "rationale": "Non-JSON fallback"
        }


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


def finish_and_summarize(final_line: str | None = None):
    """Emit final message, generate summary, and stop the app run."""
    final_message = final_line or "Thank you for answering my questions. Your history is being forwarded to your doctor!"
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    st.session_state.asked_questions.append(final_message)
    try:
        st.session_state.final_summary = generate_clinician_summary_via_model(
            st.session_state.profile,
            st.session_state.messages
        )
    except Exception:
        st.session_state.final_summary = None
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

    # 2) get JSON step
    data = call_json(user_text)

    # 3) merge structured data and capture red flags silently
    st.session_state.profile = merge_profile(st.session_state.profile, data.get("extracted_fields", {}))
    process_red_flags(data.get("red_flags", []))

    # 4) derive next question and model finish decision
    model_finish = bool(data.get("finish", False))
    missing_fields = data.get("missing_fields", [])
    st.session_state.missing_fields_latest = missing_fields
    next_q = (data.get("next_question") or "Please tell me more.").strip()

    # 5) de-dupe (avoid repeating last assistant question)
    last_assistant = next(
        (m["content"].strip() for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        ""
    )
    if last_assistant and next_q.lower() == last_assistant.lower():
        try:
            data2 = regenerate_advance(user_text, last_assistant)
            st.session_state.profile = merge_profile(st.session_state.profile, data2.get("extracted_fields", {}))
            process_red_flags(data2.get("red_flags", []))
            next_q = (data2.get("next_question") or "Thanks — please add one new detail.").strip()
            model_finish = bool(data2.get("finish", model_finish))
            st.session_state.missing_fields_latest = data2.get("missing_fields", missing_fields)
        except Exception:
            next_q = "Thanks — please add one new detail."

    # 6) if model announced completion or we hit the cap, finish
    if model_finish or st.session_state.q_count >= MAX_QUESTIONS:
        if next_q.lower().startswith("thank you for answering my questions"):
            # use model's final line if provided
            finish_and_summarize(final_line=next_q)
        else:
            finish_and_summarize()

    # 7) otherwise continue the interview
    st.session_state.messages.append({"role": "assistant", "content": next_q})
    st.session_state.asked_questions.append(next_q)
    st.session_state.q_count += 1

    # 8) hard stop if we just reached the cap
    if st.session_state.q_count >= MAX_QUESTIONS:
        finish_and_summarize()

    st.rerun()

# ─────────────────────────────────────────────
# Sidebar – actions & progress
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
        except Exception:
            st.warning("Summary generation failed. Please try again.")

    st.markdown(f"**Questions asked:** {st.session_state.q_count} / {MAX_QUESTIONS}")

    st.header("Progress")
    st.json({"missing_fields_latest": st.session_state.missing_fields_latest}, expanded=False)

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.profile = {
            "demographics": {},
            "chief_complaint": None,
            "modules": {},
            "medications": [],
            "allergies": [],
            "past_medical_history": {},
            "family_history": {},
            "social_history": {},
            "red_flags": [],
            "red_flags_checked": False,
            "free_text_notes": []
        }
        st.session_state.asked_questions = []
        st.session_state.final_summary = None
        st.session_state.q_count = 0
        st.session_state.missing_fields_latest = []
        st.rerun()

# clinical_assistant_demo.py — Patient/Doctor split + doctor-linked QR + auto-routed summaries + transcript save
import json
import time
import io
import uuid
import hashlib
import hmac
from urllib.parse import urlencode

import streamlit as st
from openai import OpenAI

# Optional QR lib (graceful fallback if not installed)
try:
    import qrcode
    _HAVE_QR = True
except ModuleNotFoundError:
    qrcode = None
    _HAVE_QR = False

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clinical Intake (Demo)", layout="centered")
client = OpenAI()  # key already configured in your env/secrets

# ─────────────────────────────────────────────
# Auth + doctor registry + inbox
# ─────────────────────────────────────────────
AUTH = st.secrets.get("auth", {})
APP_BASE_URL = (AUTH.get("app_base_url") or "").rstrip("/")
if not APP_BASE_URL:
    st.warning(
        "QR will use http://localhost:8501 since [auth].app_base_url is not set in secrets. "
        "Set it to your deployed host to share QR across devices."
    )
    APP_BASE_URL = "http://localhost:8501"
AUTH_SALT = AUTH.get("salt", "change-this-salt")
USERS = AUTH.get("users", [])

def _sha256(pw: str, salt: str) -> str:
    return hashlib.sha256((pw + salt).encode("utf-8")).hexdigest()

def _verify_pw(username: str, password: str):
    for u in USERS:
        if u.get("username") == username:
            expected = str(u.get("password_sha256", "")).strip()
            got = _sha256(password, AUTH_SALT)
            if hmac.compare_digest(expected, got):
                return {"username": username, "display_name": u.get("display_name", username)}
    if not USERS and username == "demo" and password == "demo":
        return {"username": "demo", "display_name": "Demo Doctor"}
    return None

def require_login():
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]
    st.title("📋 Doctor Dashboard")
    st.header("🔐 Login")
    with st.form("login_form"):
        u = st.text_input("Username", value="demo" if not USERS else "")
        p = st.text_input("Password", type="password", value="demo" if not USERS else "")
        ok = st.form_submit_button("Sign in")
    if ok:
        user = _verify_pw(u, p)
        if user:
            st.session_state["auth_user"] = user
            st.success(f"Welcome, {user['display_name']}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()

@st.cache_resource
def doctor_store():
    store = {}
    if USERS:
        for u in USERS:
            did = f"doc_{u['username']}"
            store[did] = {"username": u["username"], "display_name": u.get("display_name", u["username"])}
    else:
        store["doc_demo"] = {"username": "demo", "display_name": "Demo Doctor"}
    return store

@st.cache_resource
def inbox_store():
    return {}

def ensure_inbox(doctor_id: str):
    inbox = inbox_store()
    inbox.setdefault(doctor_id, [])
    return inbox[doctor_id]

def save_encounter(doctor_id: str, profile: dict, summary_md: str, transcript: list):
    enc = {
        "encounter_id": uuid.uuid4().hex,
        "created_ts": int(time.time()),
        "profile": profile,
        "summary_md": summary_md,
        "transcript": transcript,
    }
    ensure_inbox(doctor_id).append(enc)
    return enc["encounter_id"]

def doctor_id_for_user(user) -> str | None:
    uname = user.get("username") if user else None
    for did, meta in doctor_store().items():
        if meta["username"] == uname:
            return did
    return None

def make_patient_link(doctor_id: str) -> str:
    params = urlencode({"mode": "patient", "doc": doctor_id})
    return f"{APP_BASE_URL}/?{params}"

def _make_qr_png(data_url: str):
    if not _HAVE_QR:
        return None
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(data_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# URL params
# ─────────────────────────────────────────────
try:
    qp = st.query_params  # Streamlit ≥1.32
    _new_qp = True
except Exception:
    qp = st.experimental_get_query_params()
    _new_qp = False

def _qp_get(key: str, default: str = "") -> str:
    raw = qp.get(key) if _new_qp else qp.get(key, [default])
    if isinstance(raw, (list, tuple)):
        return raw[0] if raw else default
    return str(raw or default)

APP_MODE  = (_qp_get("mode") or "").lower()
DOCTOR_ID = _qp_get("doc") or None

# ─────────────────────────────────────────────
# Patient state
# ─────────────────────────────────────────────
def init_patient_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
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
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = []
    if "final_summary" not in st.session_state:
        st.session_state.final_summary = None
    if "q_count" not in st.session_state:
        st.session_state.q_count = 0

MAX_QUESTIONS = 30

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
# System prompts
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
- next_question: string
- extracted_fields: object
- red_flags: string[]
- rationale: string

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

SUMMARY_TASK = """
You are assisting a clinician.

TASK:
- Write a concise clinical summary (max 150 words).
- Highlight red flags.
- Provide up to 4 differential diagnoses ranked from most to least likely.
- For the most likely differential, list key history or exam findings that support it.
- Recommend next steps and first-line treatment.
- For each recommended diagnostic test, briefly explain the rationale.
- Under 'Consult Suggestions', include key questions (with why) and key examinations (with why).
- Use bullet points, ranked lists, and concise medical clarity. No disclaimers.

INPUTS:
- Structured patient profile (JSON) with demographics, chief complaint, modules, medications, allergies, PMHx, FHx, SHx, red flags, notes.
- Conversation transcript (user/assistant turns). Prefer structured data as source of truth.

OUTPUT:
Return plain markdown bullets — no preamble, no code fences.
"""

# (helper functions, run_patient_mode, run_doctor_dashboard, and routing go here — unchanged from the last block I gave you, including the skip-QR button inside run_doctor_dashboard)

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
st.caption("Please tell me your chief presenting complaint.")

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
Goal: take a concise, safe medical


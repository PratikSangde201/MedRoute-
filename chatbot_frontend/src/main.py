import json
import os
import time
from uuid import uuid4

import requests
import streamlit as st

API_BASE_URL = os.getenv("CHATBOT_API_BASE_URL", "http://localhost:8000")
CHATBOT_URL = os.getenv("CHATBOT_URL", f"{API_BASE_URL}/chatbot-rag-agent")

st.set_page_config(
    page_title="MedRoute",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] > section.main,
[data-testid="stMain"],
.block-container {
    background-color: #C2D6EC !important;
}
[data-testid="stSidebar"] {
    border-right: 3px solid #1E40AF;
}
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div,
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #C2D6EC !important;
    border-top: none !important;
    box-shadow: none !important;
}
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    box-shadow: 0 2px 8px rgba(30, 64, 175, 0.12) !important;
    border: 1px solid #BAD0E8 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 4px solid #1E40AF !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 4px solid #15803D !important;
    background-color: #F0FDF4 !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #0A1628 !important;
    font-size: 15px !important;
}
div.stButton > button[kind="primary"],
div.stButton > button {
    background-color: #1E40AF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
div.stButton > button:hover {
    background-color: #1D4ED8 !important;
}
.route-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-FACTUAL    { background:#DCFCE7; color:#14532D; border:1px solid #86EFAC; }
.badge-RELATIONAL { background:#FEF3C7; color:#713F12; border:1px solid #FCD34D; }
.badge-COMPLEX    { background:#EDE9FE; color:#3B0764; border:1px solid #C4B5FD; }
.badge-GENERAL    { background:#DBEAFE; color:#1E3A8A; border:1px solid #93C5FD; }
.badge-HYBRID     { background:#F1F5F9; color:#334155; border:1px solid #CBD5E1; }
.badge-TIMEOUT    { background:#FEE2E2; color:#7F1D1D; border:1px solid #FCA5A5; }
</style>
""", unsafe_allow_html=True)

_ROUTE_LABELS = {
    "FACTUAL":    ("FACTUAL",    "badge-FACTUAL"),
    "RELATIONAL": ("RELATIONAL", "badge-RELATIONAL"),
    "COMPLEX":    ("COMPLEX",    "badge-COMPLEX"),
    "GENERAL":    ("GENERAL",    "badge-GENERAL"),
    "HYBRID":     ("HYBRID",     "badge-HYBRID"),
    "TIMEOUT":    ("TIMEOUT",    "badge-TIMEOUT"),
}


def is_unavailable_answer(text: str) -> bool:
    lowered = (text or "").lower()
    return any(m in lowered for m in [
        "i'm sorry, but i don't have", "don't have any information",
        "no information", "insufficient information",
        "currently unavailable", "no retriever context found",
    ])


def request_with_retry(method, url, retries=3, backoff_seconds=1.5, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return requests.request(method=method, url=url, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds)
    raise last_exc


def build_chat_history_text(messages: list[dict], max_messages: int = 8) -> str:
    if not messages:
        return ""
    lines = []
    for item in messages[-max_messages:]:
        role = item.get("role", "user")
        content = item.get("output", "")
        if content:
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(lines)


def _route_badge_html(route: str) -> str:
    label, css = _ROUTE_LABELS.get(route, (route, "badge-HYBRID"))
    return f'<span class="route-badge {css}">{label}</span>'


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

def render_chat_page():
    st.markdown("### Medical Query Assistant — MedRoute")
    st.caption("Ask about diseases, symptoms, treatments, precautions, or general health advice.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            route = message.get("route", "")
            if message["role"] == "assistant" and route:
                st.markdown(_route_badge_html(route), unsafe_allow_html=True)
            if "output" in message:
                st.markdown(message["output"])
            srcs = message.get("sources", [])
            if srcs and message["role"] == "assistant":
                with st.expander("Sources", expanded=False):
                    for src in srcs[:3]:
                        if isinstance(src, dict):
                            label = src.get("metadata", {}).get("source", "source")
                            content = src.get("content", "")[:220]
                            st.caption(f"**{label}** — {content}")

    if prompt := st.chat_input("Ask a medical question…"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({
            "message_id": str(uuid4()), "role": "user", "output": prompt,
        })
        chat_history = build_chat_history_text(st.session_state.messages[:-1])

        response = None
        with st.spinner("Searching…"):
            try:
                response = request_with_retry(
                    "POST", CHATBOT_URL, retries=2, backoff_seconds=2.0,
                    json={
                        "text": prompt, "query": prompt,
                        "chat_history": chat_history,
                        "routing": st.session_state.get("routing_enabled", True),
                    },
                    timeout=100,
                )
            except requests.exceptions.RequestException:
                response = None

        if response is not None and response.status_code == 200:
            payload = response.json()
            output_text = payload.get("output", "")
            sources = payload.get("sources", [])
            route = payload.get("route", "")
            if is_unavailable_answer(output_text):
                sources = []
        elif response is None:
            output_text = "Backend is temporarily unreachable. Please wait and try again."
            sources, route = [], ""
        else:
            output_text = f"Error {response.status_code}: {response.text[:200]}"
            sources, route = [], ""

        with st.chat_message("assistant"):
            if route:
                st.markdown(_route_badge_html(route), unsafe_allow_html=True)
            st.markdown(output_text)
            if sources:
                with st.expander("Sources", expanded=False):
                    for src in sources[:3]:
                        if isinstance(src, dict):
                            label = src.get("metadata", {}).get("source", "source")
                            content = src.get("content", "")[:220]
                            st.caption(f"**{label}** — {content}")

        st.session_state.messages.append({
            "message_id": str(uuid4()), "role": "assistant",
            "output": output_text, "sources": sources, "route": route,
        })


# ---------------------------------------------------------------------------
# Ingest page — abstract status, no raw job IDs or tables
# ---------------------------------------------------------------------------

def render_ingest_page():
    st.markdown("### Ingest Medical Content")
    st.caption("Upload a file to add new medical knowledge to the system.")

    if "active_job_id" not in st.session_state:
        st.session_state.active_job_id = ""

    # ── Upload section ───────────────────────────────────────────────────────
    if not st.session_state.active_job_id:
        uploaded = st.file_uploader(
            "Choose a file", type=["pdf", "txt", "csv", "md"],
            help="Supported: PDF, TXT, CSV, Markdown",
        )
        if st.button("Start Ingestion", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Please choose a file first.")
            else:
                with st.spinner("Uploading…"):
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/ingest/upload",
                            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                            timeout=120,
                        )
                        if r.status_code == 200:
                            st.session_state.active_job_id = r.json()["job_id"]
                            st.rerun()
                        else:
                            st.error("Upload failed. Please try again.")
                    except Exception:
                        st.error("Could not reach the server. Is the backend running?")
        return

    # ── Poll status abstractly ───────────────────────────────────────────────
    try:
        status_r = requests.get(
            f"{API_BASE_URL}/ingest/status/{st.session_state.active_job_id}", timeout=30
        )
        if status_r.status_code != 200:
            st.error("Could not fetch processing status.")
            if st.button("Start over"):
                st.session_state.active_job_id = ""
                st.rerun()
            return
        job = status_r.json()
    except Exception:
        st.error("Server unreachable while checking status.")
        return

    status = job.get("status", "unknown")

    # ── Processing in progress ───────────────────────────────────────────────
    if status in {"pending", "processing", "inserting"}:
        stage_info = {
            "pending":    ("Reading file content…",         15),
            "processing": ("Analysing and extracting data…", 60),
            "inserting":  ("Writing to knowledge base…",     88),
        }
        label, pct = stage_info.get(status, ("Processing…", 50))
        st.progress(pct / 100, text=label)
        st.caption("This usually takes 15–60 seconds. The page refreshes automatically.")
        time.sleep(3)
        st.rerun()

    # ── Ready for review ─────────────────────────────────────────────────────
    elif status == "review_needed":
        st.success("Analysis complete — please review the extracted information below.")

        confidence = job.get("confidence_score")
        if confidence is not None:
            quality = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
            col_m, _ = st.columns([1, 3])
            with col_m:
                st.metric("Extraction Quality", quality, f"{confidence:.0%} confidence")

        validation_result = job.get("validation_result") or {}
        if validation_result.get("errors"):
            st.error("Errors:\n- " + "\n- ".join(validation_result["errors"]))
        if validation_result.get("warnings"):
            st.warning("Warnings:\n- " + "\n- ".join(validation_result["warnings"]))

        structured = job.get("structured") or {"diseases": []}
        structured_text = st.text_area(
            "Extracted data (editable JSON)",
            value=json.dumps(structured, indent=2),
            height=300,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve & Save", type="primary", use_container_width=True):
                try:
                    parsed = json.loads(structured_text)
                except Exception:
                    st.error("Invalid JSON — fix it before approving.")
                    return
                r = requests.post(
                    f"{API_BASE_URL}/ingest/approve/{st.session_state.active_job_id}",
                    json={"structured": parsed}, timeout=120,
                )
                if r.status_code == 200:
                    st.session_state.active_job_id = ""
                    st.rerun()
                else:
                    st.error("Approval failed.")
        with col2:
            if st.button("Discard", use_container_width=True):
                requests.post(
                    f"{API_BASE_URL}/ingest/reject/{st.session_state.active_job_id}",
                    timeout=30,
                )
                st.session_state.active_job_id = ""
                st.rerun()

    # ── Terminal states ───────────────────────────────────────────────────────
    elif status == "approved":
        st.success("Knowledge successfully added to the system.")
        if st.button("Upload another file", use_container_width=True):
            st.session_state.active_job_id = ""
            st.rerun()

    elif status in {"rejected", "discarded"}:
        st.info("Submission was discarded.")
        if st.button("Start over", use_container_width=True):
            st.session_state.active_job_id = ""
            st.rerun()

    elif status == "failed":
        st.error(f"Processing failed: {job.get('error_message') or 'Unknown error'}")
        if st.button("Try again", use_container_width=True):
            st.session_state.active_job_id = ""
            st.rerun()

    else:
        st.warning(f"Unexpected status: {status}")
        if st.button("Reset", use_container_width=True):
            st.session_state.active_job_id = ""
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — title + 2 nav options only, nothing else
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## MedRoute")

    page = st.radio(
        "Navigation",
        ["Chat", "Ingest & Review"],
        index=0,
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------

if page == "Chat":
    render_chat_page()
else:
    render_ingest_page()

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
/* ── Base ───────────────────────────────────────────────────────────────── */
html, body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ── Main content ───────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > section.main,
[data-testid="stMain"],
.block-container {
    background-color: #EBF5EE !important;
    color: #111827 !important;
}
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 1rem !important;
    max-width: 860px !important;
}

/* Force all text dark in main area */
.block-container p,
.block-container span,
.block-container li,
.block-container h1,
.block-container h2,
.block-container h3,
.block-container label {
    color: #111827 !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #14532D !important;
    border-right: 2px solid #166534 !important;
}
/* Kill white backgrounds on every inner wrapper Streamlit generates */
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stRadio > div,
[data-testid="stSidebar"] [class*="block-container"],
[data-testid="stSidebar"] [class*="st-emotion-cache"] {
    background-color: transparent !important;
    background: transparent !important;
}
/* Force every text node in the sidebar to light green — span catches radio labels */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] em,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #ECFDF5 !important;
}
/* Radio group heading */
[data-testid="stSidebar"] .stRadio > label > div > p {
    color: #A7F3D0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Chat input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div,
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #EBF5EE !important;
    border-top: 2px solid #6EE7B7 !important;
    box-shadow: 0 -2px 8px rgba(5,150,105,0.08) !important;
}
/* Border and shadow on the wrapper so the whole input box is clearly framed */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 2.5px solid #059669 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.18) !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    font-size: 15px !important;
    border: none !important;
    outline: none !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6B7280 !important;
}

/* ── Chat messages ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07) !important;
    border: 1px solid #BBF7D0 !important;
    color: #111827 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #111827 !important;
}
/* Assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF !important;
    border-left: 4px solid #059669 !important;
}
/* User */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #D1FAE5 !important;
    border-left: 4px solid #065F46 !important;
}

/* ── Expander (Sources) ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #F0FFF4 !important;
    border: 1px solid #A7F3D0 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary span {
    color: #065F46 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] p {
    color: #374151 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
div.stButton > button {
    background-color: #059669 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
div.stButton > button:hover {
    background-color: #047857 !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #F0FFF4 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: #111827 !important;
}

/* ── Text area (JSON edit) ──────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid #A7F3D0 !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 13px !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] p {
    color: #111827 !important;
}

/* ── Metric ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] label,
[data-testid="stMetricValue"] {
    color: #111827 !important;
}

/* ── Caption / small text ───────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #374151 !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    color: #059669 !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
    border-color: #A7F3D0 !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background-color: #059669 !important;
}

/* ── Route badges ───────────────────────────────────────────────────────── */
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
.badge-FACTUAL    { background:#D1FAE5; color:#065F46; border:1px solid #6EE7B7; }
.badge-RELATIONAL { background:#FEF3C7; color:#78350F; border:1px solid #FCD34D; }
.badge-COMPLEX    { background:#EDE9FE; color:#4C1D95; border:1px solid #A78BFA; }
.badge-GENERAL    { background:#E0F2FE; color:#0C4A6E; border:1px solid #7DD3FC; }
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
    st.markdown("## MedRoute — Medical Query Assistant")
    st.caption(
        "Ask about diseases, symptoms, treatments, or precautions. "
        "Responses are drawn from a structured medical knowledge graph and document retrieval."
    )
    st.divider()

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
        with st.spinner("Searching knowledge base…"):
            try:
                response = request_with_retry(
                    "POST", CHATBOT_URL, retries=2, backoff_seconds=2.0,
                    json={
                        "text": prompt, "query": prompt,
                        "chat_history": chat_history,
                        "routing": True,
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
# Ingest page
# ---------------------------------------------------------------------------

def render_ingest_page():
    st.markdown("## Ingest & Manage Medical Content")
    st.caption("Upload files to add knowledge, or manage existing diseases in the graph.")
    st.divider()

    tab_upload, tab_manage = st.tabs(["Upload & Review", "Manage Knowledge"])

    with tab_manage:
        _render_manage_tab()

    with tab_upload:
        _render_upload_tab()


def _render_manage_tab():
    st.markdown("#### Knowledge Graph — Disease Index")
    st.caption("All diseases currently stored in Neo4j. You can delete any entry and its orphaned symptoms/precautions.")

    if st.button("Refresh list", use_container_width=False):
        st.session_state.pop("disease_list", None)

    if "disease_list" not in st.session_state:
        with st.spinner("Loading diseases from graph…"):
            try:
                r = requests.get(f"{API_BASE_URL}/ingest/diseases", timeout=15)
                st.session_state.disease_list = r.json().get("diseases", []) if r.status_code == 200 else []
            except Exception:
                st.session_state.disease_list = []

    diseases = st.session_state.get("disease_list", [])

    if not diseases:
        st.info("No diseases found in the knowledge graph.")
        return

    st.caption(f"{len(diseases)} disease(s) in graph")

    # Confirm-before-delete state
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = ""

    for d in diseases:
        name = d["name"]
        syms = d["symptoms"]
        precs = d["precautions"]
        col_name, col_stats, col_btn = st.columns([4, 3, 2])
        with col_name:
            st.markdown(f"**{name}**")
        with col_stats:
            st.caption(f"{syms} symptom(s) · {precs} precaution(s)")
        with col_btn:
            if st.session_state.confirm_delete == name:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, delete", key=f"yes_{name}", use_container_width=True):
                        try:
                            r = requests.delete(
                                f"{API_BASE_URL}/ingest/disease/{requests.utils.quote(name, safe='')}",
                                timeout=15,
                            )
                            if r.status_code == 200:
                                st.success(f"'{name}' deleted.")
                            else:
                                st.error(f"Delete failed: {r.text[:120]}")
                        except Exception as exc:
                            st.error(f"Error: {exc}")
                        st.session_state.confirm_delete = ""
                        st.session_state.pop("disease_list", None)
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_{name}", use_container_width=True):
                        st.session_state.confirm_delete = ""
                        st.rerun()
            else:
                if st.button("Delete", key=f"del_{name}", use_container_width=True):
                    st.session_state.confirm_delete = name
                    st.rerun()


def _render_upload_tab():
    if "active_job_id" not in st.session_state:
        st.session_state.active_job_id = ""

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

    if status in {"pending", "processing", "inserting"}:
        stage_info = {
            "pending":    ("Reading file content…",          15),
            "processing": ("Analysing and extracting data…", 60),
            "inserting":  ("Writing to knowledge base…",     88),
        }
        label, pct = stage_info.get(status, ("Processing…", 50))
        st.progress(pct / 100, text=label)
        st.caption("This usually takes 15–60 seconds. The page refreshes automatically.")
        time.sleep(3)
        st.rerun()

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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## MedRoute")
    st.markdown(
        "<p style='color:#A7F3D0; font-size:0.85rem; line-height:1.6; margin-bottom:0.6rem;'>"
        "AI-powered medical assistant backed by a Neo4j knowledge graph. "
        "Ask about diseases, symptoms, treatments, and precautions."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

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

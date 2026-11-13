import os
import re
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Media Intelligence Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* 1. Outer Page Background ONLY: Pure White */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #FFFFFF !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* 2. Sidebar: Dark Black (#090D16) with White Text */
        section[data-testid="stSidebar"] {
            background-color: #090D16 !important;
            border-right: 1px solid #1E293B !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            color: #94A3B8 !important;
        }

        /* 3. White Text Fields, TextAreas & Language Dropdown */
        .stTextArea textarea,
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 10px !important;
            caret-color: #2563EB !important;
            font-weight: 500 !important;
        }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.35) !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #64748B !important;
        }

        /* Selectbox (Language Dropdown) */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] svg {
            color: #0F172A !important;
            fill: #0F172A !important;
        }
        div[data-baseweb="popover"] ul {
            background-color: #FFFFFF !important;
        }
        div[data-baseweb="popover"] li {
            color: #0F172A !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #EFF6FF !important;
        }

        /* File Uploader */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF !important;
            border: 1.5px dashed #94A3B8 !important;
            border-radius: 10px !important;
        }
        section[data-testid="stFileUploaderDropzone"] * {
            color: #0F172A !important;
        }

        /* 4. All Buttons: Royal Blue with White Text */
        .stButton > button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button p,
        .stButton > button span {
            color: #FFFFFF !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            transform: translateY(-1px);
        }

        /* Secondary / Inactive Session Buttons */
        .stButton > button[kind="secondary"] {
            background: #111827 !important;
            border: 1px solid #2563EB !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background: #1E3A8A !important;
        }

        /* 5. Top Header Bar */
        .studio-header {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 16px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }
        .studio-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #FFFFFF !important;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .studio-sub {
            font-size: 0.85rem;
            color: #94A3B8 !important;
            margin: 0.2rem 0 0 0;
        }
        .studio-pill {
            background-color: rgba(37, 99, 235, 0.2);
            color: #60A5FA !important;
            border: 1px solid #2563EB;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        /* 6. Royal Blue Hero Card */
        .hero-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            border-radius: 20px;
            padding: 1.6rem;
            color: #FFFFFF !important;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.25);
            height: 100%;
        }
        .hero-tag {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #BFDBFE !important;
            margin-bottom: 0.4rem;
        }
        .hero-heading {
            font-size: 1.35rem;
            font-weight: 700;
            color: #FFFFFF !important;
            line-height: 1.3;
            margin-bottom: 0.35rem;
        }
        .hero-url {
            font-size: 0.8rem;
            color: #DBEAFE !important;
            margin-bottom: 1.25rem;
            word-break: break-all;
        }
        .hero-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem;
        }
        .stat-cyan {
            background-color: #7DD3FC;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
        }
        .stat-cyan .num { font-size: 1.35rem; font-weight: 800; color: #0C4A6E !important; }
        .stat-cyan .lbl { font-size: 0.78rem; font-weight: 600; color: #0369A1 !important; }

        .stat-peach {
            background-color: #FDBA74;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
        }
        .stat-peach .num { font-size: 1.35rem; font-weight: 800; color: #7C2D12 !important; }
        .stat-peach .lbl { font-size: 0.78rem; font-weight: 600; color: #9A3412 !important; }

        /* 7. 4 Vibrant Metric Tiles */
        .tiles-wrapper {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.9rem;
            height: 100%;
        }
        .cat-tile {
            border-radius: 18px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
        }
        .bg-emerald { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
        .bg-coral   { background: linear-gradient(135deg, #F97316 0%, #EA580C 100%); }
        .bg-amber   { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
        .bg-royal   { background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); }

        .cat-icon {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.25);
            color: #FFFFFF !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
        .cat-name { font-size: 0.98rem; font-weight: 700; color: #FFFFFF !important; }
        .cat-count { font-size: 0.8rem; color: rgba(255, 255, 255, 0.9) !important; margin-top: 0.15rem; }

        /* 8. Dark Slate Sequential Cards & Feed Rows */
        .content-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            color: #FFFFFF !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }
        .content-card p, .content-card li, .content-card h1, .content-card h2, .content-card h3 {
            color: #FFFFFF !important;
        }
        .card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }
        .card-top-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF !important;
        }
        .step-badge {
            background-color: #2563EB;
            color: #FFFFFF !important;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            margin-right: 0.6rem;
            letter-spacing: 0.04em;
        }
        .transcript-scroll {
            max-height: 260px;
            overflow-y: auto;
            background-color: #090D16;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: #E2E8F0 !important;
            font-size: 0.92rem;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .row-item {
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid #1E293B;
        }
        .row-item:last-child { border-bottom: none; }
        .row-badge {
            min-width: 32px;
            height: 32px;
            border-radius: 50%;
            color: #FFFFFF !important;
            font-size: 0.75rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .row-body {
            font-size: 0.92rem;
            color: #F8FAFC !important;
            line-height: 1.5;
            padding-top: 0.25rem;
        }

        /* 9. Chat Messages */
        div[data-testid="stChatMessage"] {
            background-color: #111827 !important;
            border: 1px solid #1E293B !important;
            border-radius: 14px !important;
            padding: 1rem !important;
        }
        div[data-testid="stChatMessage"] * {
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def parse_items(items) -> list:
    if not items:
        return []
    if isinstance(items, list):
        return [str(i).strip() for i in items if str(i).strip()]
    return [line.strip("-•* ").strip() for line in str(items).splitlines() if line.strip()]


def clean_currency(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"(?<!\\)\$", r"\\$", str(text))


def build_rows_html(items: list, color_hex: str, prefix: str) -> str:
    if not items:
        return "<div style='color:#94A3B8; font-size:0.9rem;'>No items extracted.</div>"
    html = ""
    for i, item in enumerate(items, 1):
        html += f"""
        <div class="row-item">
            <div class="row-badge" style="background-color:{color_hex};">{prefix}{i}</div>
            <div class="row-body">{item}</div>
        </div>
        """
    return html


def run_pipeline_progressive(source: str, source_label: str, language: str) -> dict:
    """Runs each stage sequentially and reveals each section on screen as soon as it finishes."""
    progress_box = st.empty()
    transcript_slot = st.empty()
    summary_slot = st.empty()
    insights_slot = st.empty()

    with progress_box.status(f"Processing: {source_label}", expanded=True) as status:
        st.write("Step 1 of 6 — Downloading and chunking audio")
        chunks = process_input(source)

        st.write(f"Step 2 of 6 — Transcribing audio ({language.capitalize()})")
        transcript = transcribe_all(chunks, language)
        if not transcript or not transcript.strip():
            status.update(label="Transcription failed", state="error")
            raise ValueError("Empty transcript returned.")

        # Immediately reveal Step 1: Transcription
        transcript_slot.markdown(
            f"""
            <div class="content-card">
                <div class="card-top">
                    <div><span class="step-badge">STEP 1</span><span class="card-top-title">Full Transcription</span></div>
                    <span class="studio-pill">{len(transcript.split()):,} WORDS</span>
                </div>
                <div class="transcript-scroll">{transcript}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Step 3 of 6 — Generating title")
        title = generate_title(transcript)

        st.write("Step 4 of 6 — Synthesizing executive summary")
        summary = summarize(transcript)

        # Immediately reveal Step 2: Executive Summary
        summary_slot.markdown(
            f"""
            <div class="content-card">
                <div class="card-top">
                    <div><span class="step-badge">STEP 2</span><span class="card-top-title">Executive Summary</span></div>
                    <span class="studio-pill">{language.upper()}</span>
                </div>
                <div style="color:#F8FAFC; line-height:1.65; font-size:0.95rem;">
                    {clean_currency(summary).replace(chr(10), '<br>')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Step 5 of 6 — Extracting action items, decisions, and open questions")
        action_items = parse_items(extract_action_items(transcript))
        decisions = parse_items(extract_key_decisions(transcript))
        questions = parse_items(extract_questions(transcript))

        # Immediately reveal Step 3: Action Items, Decisions & Questions
        with insights_slot.container():
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(
                    f"""
                    <div class="content-card">
                        <div class="card-top">
                            <div><span class="step-badge">STEP 3A</span><span class="card-top-title">Action Items</span></div>
                            <span class="studio-pill">{len(action_items)}</span>
                        </div>
                        {build_rows_html(action_items, '#10B981', 'A')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="content-card">
                        <div class="card-top">
                            <div><span class="step-badge">STEP 3B</span><span class="card-top-title">Key Decisions</span></div>
                            <span class="studio-pill">{len(decisions)}</span>
                        </div>
                        {build_rows_html(decisions, '#2563EB', 'D')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="content-card">
                        <div class="card-top">
                            <div><span class="step-badge">STEP 3C</span><span class="card-top-title">Open Questions</span></div>
                            <span class="studio-pill">{len(questions)}</span>
                        </div>
                        {build_rows_html(questions, '#F97316', 'Q')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("Step 6 of 6 — Indexing transcript for Chatbot Q&A")
        rag_chain = build_rag_chain(transcript)

        status.update(label=f"Completed: {title}", state="complete", expanded=False)

    return {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "source_label": source_label,
        "language": language,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "chat_history": [],
    }


# --- SIDEBAR: DASHBOARD ---
with st.sidebar:
    st.markdown("### Dashboard")
    st.caption("Add single or multiple YouTube links to analyze and switch between sessions.")

    input_mode = st.radio(
        "Source Type",
        ["YouTube URL(s)", "Local Media File(s)"],
        label_visibility="collapsed",
    )

    yt_urls_raw = ""
    uploaded_files = None

    if input_mode == "YouTube URL(s)":
        yt_urls_raw = st.text_area(
            "YouTube Links (one per line)",
            placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
            height=100,
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload Audio/Video",
            type=["mp4", "mp3", "wav", "m4a", "mkv", "webm"],
            accept_multiple_files=True,
        )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("Process Source", type="primary", use_container_width=True)

    if st.session_state.sessions:
        st.divider()
        st.markdown("### Saved Sessions")
        for sid, sdata in reversed(list(st.session_state.sessions.items())):
            is_active = sid == st.session_state.active_session_id
            btn_type = "primary" if is_active else "secondary"
            label = sdata["title"] if len(sdata["title"]) <= 30 else sdata["title"][:27] + "..."
            if st.button(label, key=f"sess_{sid}", type=btn_type, use_container_width=True):
                st.session_state.active_session_id = sid
                st.rerun()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if st.button("Clear All Sessions", type="secondary", use_container_width=True):
            st.session_state.sessions = {}
            st.session_state.active_session_id = None
            st.rerun()

# --- HEADER ---
session_count = len(st.session_state.sessions)
st.markdown(
    f"""
    <div class="studio-header">
        <div>
            <div class="studio-title">Media Intelligence Studio</div>
            <div class="studio-sub">Sequential transcription, structured insights, and contextual Q&A</div>
        </div>
        <div class="studio-pill">{session_count} ACTIVE SESSION{"S" if session_count != 1 else ""}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- EXECUTION ---
if run_btn:
    if input_mode == "YouTube URL(s)":
        urls = [u.strip() for u in yt_urls_raw.splitlines() if u.strip()]
        if not urls:
            st.error("Please enter at least one YouTube URL.")
        else:
            for url in urls:
                try:
                    data = run_pipeline_progressive(url, url, language)
                    st.session_state.sessions[data["id"]] = data
                    st.session_state.active_session_id = data["id"]
                except Exception as e:
                    st.error(f"Error processing {url}: {e}")
            st.rerun()
    else:
        if not uploaded_files:
            st.error("Please upload at least one audio or video file.")
        else:
            for up_file in uploaded_files:
                tmp_path = None
                try:
                    suffix = os.path.splitext(up_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up_file.read())
                        tmp_path = tmp.name
                    data = run_pipeline_progressive(tmp_path, up_file.name, language)
                    st.session_state.sessions[data["id"]] = data
                    st.session_state.active_session_id = data["id"]
                except Exception as e:
                    st.error(f"Error processing {up_file.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
            st.rerun()

# --- SEQUENTIAL DISPLAY WORKSPACE ---
active = st.session_state.sessions.get(st.session_state.active_session_id)

if active:
    n_actions = len(active["action_items"])
    n_decisions = len(active["key_decisions"])
    n_questions = len(active["open_questions"])
    n_words = len(active["transcript"].split())

    # 0. TOP OVERVIEW & METRICS
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-tag">Active Analysis</div>
                <div class="hero-heading">{active['title']}</div>
                <div class="hero-url">{active['source_label']}</div>
                <div class="hero-stats">
                    <div class="stat-cyan">
                        <div class="num">{n_words:,}</div>
                        <div class="lbl">Words Transcribed</div>
                    </div>
                    <div class="stat-peach">
                        <div class="num">{n_actions + n_decisions + n_questions}</div>
                        <div class="lbl">Extracted Insights</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="tiles-wrapper">
                <div class="cat-tile bg-emerald">
                    <div class="cat-icon">A</div>
                    <div>
                        <div class="cat-name">Action Items</div>
                        <div class="cat-count">{n_actions} items</div>
                    </div>
                </div>
                <div class="cat-tile bg-coral">
                    <div class="cat-icon">D</div>
                    <div>
                        <div class="cat-name">Decisions</div>
                        <div class="cat-count">{n_decisions} logged</div>
                    </div>
                </div>
                <div class="cat-tile bg-amber">
                    <div class="cat-icon">Q</div>
                    <div>
                        <div class="cat-name">Questions</div>
                        <div class="cat-count">{n_questions} open</div>
                    </div>
                </div>
                <div class="cat-tile bg-royal">
                    <div class="cat-icon">L</div>
                    <div>
                        <div class="cat-name">Language</div>
                        <div class="cat-count">{active['language'].capitalize()}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # 1. STEP 1: FULL TRANSCRIPTION
    st.markdown(
        f"""
        <div class="content-card">
            <div class="card-top">
                <div><span class="step-badge">STEP 1</span><span class="card-top-title">Full Transcription</span></div>
                <span class="studio-pill">{n_words:,} WORDS</span>
            </div>
            <div class="transcript-scroll">{active['transcript']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. STEP 2: EXECUTIVE SUMMARY
    st.markdown(
        f"""
        <div class="content-card">
            <div class="card-top">
                <div><span class="step-badge">STEP 2</span><span class="card-top-title">Executive Summary</span></div>
                <span class="studio-pill">{active['language'].upper()}</span>
            </div>
            <div style="color:#F8FAFC; line-height:1.65; font-size:0.95rem;">
                {clean_currency(active['summary']).replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. STEP 3: ACTION ITEMS, KEY DECISIONS & OPEN QUESTIONS
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="card-top">
                    <div><span class="step-badge">STEP 3A</span><span class="card-top-title">Action Items</span></div>
                    <span class="studio-pill">{n_actions}</span>
                </div>
                {build_rows_html(active['action_items'], '#10B981', 'A')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="card-top">
                    <div><span class="step-badge">STEP 3B</span><span class="card-top-title">Key Decisions</span></div>
                    <span class="studio-pill">{n_decisions}</span>
                </div>
                {build_rows_html(active['key_decisions'], '#2563EB', 'D')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="card-top">
                    <div><span class="step-badge">STEP 3C</span><span class="card-top-title">Open Questions</span></div>
                    <span class="studio-pill">{n_questions}</span>
                </div>
                {build_rows_html(active['open_questions'], '#F97316', 'Q')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. STEP 4: INTERACTIVE CHATBOT (AT THE VERY BOTTOM)
    st.markdown(
        """
        <div class="content-card" style="margin-top:0.5rem;">
            <div class="card-top" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                <div><span class="step-badge">STEP 4</span><span class="card-top-title">Interactive Q&A Assistant</span></div>
                <span class="studio-pill">CHATBOT READY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in active["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(clean_currency(msg["content"]))

    if prompt := st.chat_input("Ask any question about this transcript..."):
        active["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching transcript..."):
                try:
                    reply = ask_question(active["rag_chain"], prompt)
                except Exception as e:
                    reply = f"Error: {e}"
                st.markdown(clean_currency(reply))

        active["chat_history"].append({"role": "assistant", "content": reply})

else:
    st.markdown(
        """
        <div class="content-card" style="text-align:center; padding:4rem 2rem;">
            <div style="width:48px; height:48px; border-radius:14px; background:rgba(37, 99, 235, 0.2); border:1px solid #2563EB; color:#60A5FA; font-weight:800; font-size:1.1rem; display:flex; align-items:center; justify-content:center; margin:0 auto 1rem auto;">
                +
            </div>
            <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF; margin-bottom:0.4rem;">
                Ready for Media Ingestion
            </div>
            <div style="font-size:0.92rem; color:#94A3B8; max-width:440px; margin:0 auto;">
                Enter one or more YouTube URLs or upload audio/video files in the Dashboard sidebar to generate transcription, summary, insights, and interactive Q&A in sequence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
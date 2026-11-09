"""
Nova - a friendly AI teacher powered by Gemini, grounded in your knowledge base (KB).

Run:
    pip install streamlit google-genai
    streamlit run app.py

API key (pick one):
    1. .streamlit/secrets.toml  ->  GEMINI_API_KEY = "your_key"
    2. environment variable     ->  export GEMINI_API_KEY="your-key"
    3. paste it in the sidebar while the app is running
"""
import streamlit as st
import io
from PIL import Image

icon_path = os.path.join(os.path.dirname(__file__), "moon.png")
icon = Image.open(icon_path)

st.set_page_config(page_title="My App Name", page_icon=icon)
from gtts import gTTS
from openai import OpenAI
import os
from pathlib import Path

from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Settings you may want to change
# ----------------------------------------------------------------------------
MODEL = "gemini-3.5-flash"
KB_PATH = Path(__file__).parent / "proto-nova-7.py"   # the code reads this file
APP_NAME = "Nova"
TAGLINE = "Ask anything from the knowledge base. I'll explain it until it clicks."

TEACHER_PROMPT = """You are Nova, an expert teacher. You teach the client whatever they want
to learn, kindly and politely. If they do not understand, explain again more clearly,
with simpler words and small examples. Answer using the knowledge base below.
If a question is outside the knowledge base, say: "Sorry, I don't have that information
yet, and I will learn it soon." Be humble, welcome the client at the start, and ask for
feedback when the conversation is ending."""

SUGGESTIONS = [
    "Give me an overview of this project",
    "Explain the main functions step by step",
    "Teach me this like I'm a beginner",
    "What can I improve in this code?",
]

st.set_page_config(page_title=f"{APP_NAME} · AI Teacher", page_icon="🎓", layout="centered")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --ink: #0F1736;
  --panel: #18224D;
  --panel-2: #212D63;
  --line: #34437F;
  --text: #EEF1FF;
  --muted: #A5AEDB;
  --accent: #FFC857;
}

html, body, [class*="css"], .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp { background: var(--ink); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 780px; padding-top: 2.2rem; padding-bottom: 6rem; }

/* Hero */
.hero { padding: 0.5rem 0 1.4rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
.hero h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 800;
  font-size: clamp(2.6rem, 8vw, 4.2rem); line-height: 1; margin: 0; color: var(--text);
  letter-spacing: -0.02em;
}
.hero h1 span { color: var(--accent); }
.hero p { color: var(--muted); margin: 0.7rem 0 0 0; font-size: 1.05rem; max-width: 34rem; }

/* Chat messages */
[data-testid="stChatMessage"] {
  background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
  padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--panel-2); margin-left: 12%;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.65; }
[data-testid="stChatMessage"] code { background: var(--ink); color: var(--accent); border-radius: 6px; padding: 0.1em 0.4em; }
[data-testid="stChatMessage"] pre { background: var(--ink) !important; border: 1px solid var(--line); border-radius: 12px; }

/* Input */
[data-testid="stChatInput"] { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] button { background: var(--accent); color: var(--ink); }

/* Buttons (suggestions + sidebar) */
.stButton > button {
  background: transparent; color: var(--text); border: 1px solid var(--line);
  border-radius: 12px; text-align: left; padding: 0.7rem 0.9rem; transition: border-color .15s, background .15s;
}
.stButton > button:hover { border-color: var(--accent); background: var(--panel); color: var(--text); }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text); }
.kb-card { border: 1px solid var(--line); border-radius: 14px; padding: 0.8rem 1rem; background: var(--ink); }
.kb-card b { color: var(--accent); }
.kb-card small { color: var(--muted); }

@media (max-width: 600px) {
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Look for the key in secrets, then environment, then the sidebar box."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@st.cache_data(show_spinner=False)
def read_kb(path: str, modified: float) -> str:
    """Read the KB file. `modified` busts the cache whenever the file changes."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def build_chat(client: genai.Client, kb_text: str):
    system_instruction = f"{TEACHER_PROMPT}\n\n===== KNOWLEDGE BASE START =====\n{kb_text}\n===== KNOWLEDGE BASE END ====="
    return client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )


def stream_reply(chat, text: str):
    for chunk in chat.send_message_stream(text):
        if chunk.text:
            yield chunk.text


def queue_prompt(text: str):
    st.session_state.queued = text


def reset_chat():
    st.session_state.messages = []
    st.session_state.chat = None
    
def speak(text: str, lang: str = "en"):
    clean = text.replace("*", "").replace("#", "").replace("`", "")[:1500]
    try:
        buf = io.BytesIO()
        gTTS(text=clean, lang=lang).write_to_fp(buf)
        st.audio(buf.getvalue(), format="audio/mp3", autoplay=True)
    except Exception:
        st.caption("Voice isn't available right now.")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🎓 {APP_NAME}")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Or set GEMINI_API_KEY in secrets.")

    uploaded = st.file_uploader("Swap knowledge base (optional)", type=["py", "txt", "md", "json", "csv"])

    kb_text, kb_label = "", ""
    if uploaded is not None:
        kb_text = uploaded.getvalue().decode("utf-8", errors="replace")
        kb_label = uploaded.name
    elif KB_PATH.exists():
        kb_text = read_kb(str(KB_PATH), KB_PATH.stat().st_mtime)
        kb_label = KB_PATH.name

    if kb_text:
        st.markdown(
            f'<div class="kb-card"><b>Knowledge base loaded</b><br>{kb_label}<br>'
            f"<small>{len(kb_text):,} characters · {kb_text.count(chr(10)) + 1:,} lines</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.error(f"No knowledge base found. Put `{KB_PATH.name}` next to app.py or upload one above.")

    st.button("Start a new chat", on_click=reset_chat, use_container_width=True)

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chat", None)
st.session_state.setdefault("kb_signature", None)

# Rebuild the chat if the KB changed
signature = (kb_label, len(kb_text))
if st.session_state.kb_signature != signature:
    st.session_state.kb_signature = signature
    reset_chat()

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    f'<div class="hero"><h1>{APP_NAME}<span>.</span></h1><p>{TAGLINE}</p></div>',
    unsafe_allow_html=True,
)

if not api_key:
    st.info("Add your Gemini API key in the sidebar to start.")
    st.stop()
if not kb_text:
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        st.markdown(m["content"])

if not st.session_state.messages:
    st.markdown("**Start with a question**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        cols[i % 2].button(s, key=f"sugg_{i}", on_click=queue_prompt, args=(s,), use_container_width=True)

prompt = st.chat_input("Ask Nova something…")

audio = st.audio_input("🎤 Or ask by voice")
if audio is not None and st.session_state.get("last_audio") != audio.file_id:
    st.session_state.last_audio = audio.file_id
    try:
        oa = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = oa.audio.transcriptions.create(
            model="whisper-1", file=("speech.wav", audio.getvalue())
        ).text
    except Exception as e:
        st.error(f"Couldn't understand the recording: `{e}`")

if "queued" in st.session_state:
    prompt = st.session_state.pop("queued")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    try:
        if st.session_state.chat is None:
            client = genai.Client(api_key=api_key)
            st.session_state.chat = build_chat(client, kb_text)

        with st.chat_message("assistant", avatar="🎓"):
            reply = st.write_stream(stream_reply(st.session_state.chat, prompt))
        st.session_state.messages.append({"role": "assistant", "content": reply})
        speak(reply)
    except Exception as e:
        st.error(f"Nova couldn't answer. Check your API key and model name (`{MODEL}`), then try again.\n\n`{e}`")
        

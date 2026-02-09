import streamlit as st
import sqlite3
import re
import docx
import PyPDF2
import nltk
import numpy as np
from datetime import datetime

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

# ===============================
# SESSION STATE
# ===============================
for k, v in {
    "user": None,
    "summary": None,
    "sentences": [],
    "clean_sentences": [],
    "filename": None,
    "show_login": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# NLTK DOWNLOADS
# ===============================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

download_nltk()

# ===============================
# DATABASE
# ===============================
def get_db():
    conn = sqlite3.connect("summaries.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            dob TEXT,
            gender TEXT,
            email TEXT,
            phone TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT,
            summary TEXT,
            created_at TEXT
        )
    """)
    return conn

db = get_db()

# ===============================
# AUTH FUNCTIONS
# ===============================
def register_user(username, password, dob, gender, email, phone):
    username = username.strip().lower()
    password = password.strip()

    if not all([username, password, dob, gender, email, phone]):
        return "empty"

    cur = db.cursor()
    cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
    if cur.fetchone():
        return "exists"

    cur.execute(
        "INSERT INTO users VALUES (NULL, ?, ?, ?, ?, ?, ?)",
        (username, password, dob, gender, email, phone)
    )
    db.commit()
    return "success"

def login_user(username, password):
    cur = db.cursor()
    cur.execute(
        "SELECT 1 FROM users WHERE username=? AND password=?",
        (username.lower(), password)
    )
    return cur.fetchone() is not None

# ===============================
# REGISTRATION → LOGIN FLOW
# ===============================
# ===============================
# REGISTRATION → LOGIN FLOW
# ===============================
if not st.session_state.user:

    # -------- REGISTER PAGE --------
    if not st.session_state.show_login:
        st.title("📝 User Registration")

        ru = st.text_input("Username")
        rp = st.text_input("Password", type="password")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")

        if st.button("Register"):
            status = register_user(ru, rp, str(dob), gender, email, phone)

            if status == "success":
                st.success("✅ Registration successful. Please login.")
                st.session_state.show_login = True
                st.rerun()

            elif status == "exists":
                st.warning("⚠️ User already registered. Please login.")
                st.session_state.show_login = True
                st.rerun()

            else:
                st.error("❌ Please fill all fields")

        st.stop()

    # -------- LOGIN PAGE --------
    else:
        st.title("🔐 Login")

        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state.user = u.lower()
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

        st.stop()


# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Menu")
st.sidebar.markdown(f"👤 **{st.session_state.user}**")
menu = st.sidebar.radio("Navigation", ["Upload", "History", "Logout"])

# ===============================
# TEXT EXTRACTION (SAFE)
# ===============================
def extract_text(file):
    text = []
    name = file.name.lower()

    if name.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        except:
            st.error("❌ Failed to read PDF")
            return ""

    elif name.endswith(".docx"):
        try:
            doc = docx.Document(file)
            for p in doc.paragraphs:
                if p.text.strip():
                    text.append(p.text)
        except:
            st.error("❌ Invalid DOCX file")
            return ""

    else:
        st.error("❌ Upload only PDF or DOCX")
        return ""

    return "\n".join(text)

# ===============================
# NLP PREPROCESS
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Z\s]", "", s)
    words = word_tokenize(s)
    return " ".join(lemmatizer.lemmatize(w) for w in words if w not in stop_words)

# ===============================
# SUMMARY (TF-IDF CENTRALITY)
# ===============================
def generate_summary(sentences, clean_sentences, top_n=4):
    if not sentences:
        return ""

    tfidf = TfidfVectorizer().fit_transform(clean_sentences)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)

    scores = sim.sum(axis=1)
    ranked = sorted(
        ((scores[i], sentences[i]) for i in range(len(sentences))),
        reverse=True
    )

    return " ".join(s for _, s in ranked[:top_n])

# ===============================
# QUESTION ANSWERING
# ===============================
def answer_question(question, threshold=0.3):
    clean_q = preprocess(question)
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(st.session_state.clean_sentences + [clean_q])
    scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    idx = scores.argmax()

    return st.session_state.sentences[idx] if scores[idx] >= threshold else "No relevant answer found."

# ===============================
# UPLOAD PAGE
# ===============================
if menu == "Upload":
    st.title("📄 Upload Legal Document")
    file = st.file_uploader("Upload PDF or DOCX", ["pdf", "docx"])

    if file:
        text = extract_text(file)

        if text:
            st.session_state.sentences = sent_tokenize(text)
            st.session_state.clean_sentences = [preprocess(s) for s in st.session_state.sentences]
            st.success("Document processed successfully")

            if st.button("Generate Summary"):
                st.session_state.summary = generate_summary(
                    st.session_state.sentences,
                    st.session_state.clean_sentences
                )

                db.execute(
                    "INSERT INTO summaries VALUES (NULL, ?, ?, ?, ?)",
                    (
                        st.session_state.user,
                        file.name,
                        st.session_state.summary,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                db.commit()

            if st.session_state.summary:
                st.subheader("📌 Summary")
                st.text_area("", st.session_state.summary, height=220)

            q = st.text_input("Ask a question")
            if st.button("Get Answer"):
                st.text_area("Answer", answer_question(q), height=120)

# ===============================
# HISTORY
# ===============================
elif menu == "History":
    rows = db.execute(
        "SELECT filename, summary, created_at FROM summaries WHERE username=? ORDER BY id DESC",
        (st.session_state.user,)
    ).fetchall()

    for f, s, d in rows:
        with st.expander(f"{f} ({d})"):
            st.write(s)

# ===============================
# LOGOUT
# ===============================
elif menu == "Logout":
    st.session_state.clear()
    st.rerun()

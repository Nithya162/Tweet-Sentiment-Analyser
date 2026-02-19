import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your file locations
INDEX_PATH = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\faiss.index"
META_PATH  = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\metadata.pkl"

# Page setup
st.set_page_config(
    page_title="Tweet Sentiment Analyser",
    page_icon="🐦",
    layout="centered"
)

# Simple clean styling
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .tweet-positive {
        background-color: #0d2d1f;
        border-left: 4px solid #00c853;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #ccffdd;
        font-size: 14px;
    }
    .tweet-negative {
        background-color: #2d0d0d;
        border-left: 4px solid #ff1744;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #ffcccc;
        font-size: 14px;
    }
    .answer-box {
        background-color: #1a1d2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2e3250;
        font-size: 16px;
        color: white;
        line-height: 1.7;
        margin: 10px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# Load everything once
@st.cache_resource
def load_all_models():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    embedder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    generator = pipeline(
        "text2text-generation", model=llm_model, tokenizer=tokenizer,
        max_new_tokens=200, do_sample=False, temperature=None, top_p=None,
    )
    return index, metadata, embedder, generator


# Search similar tweets
def find_tweets(question, index, metadata, embedder):
    q_vec           = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=6)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        tweet = metadata[idx].copy()
        results.append(tweet)
    return results


# Generate answer
def generate_answer(question, tweets, generator):
    context = ""
    for t in tweets:
        context += t["sentiment"].upper() + ": " + t["clean_text"] + "\n"

    prompt = (
        "You are a helpful assistant that analyses tweet sentiments.\n"
        "Use the tweets below to answer the question clearly.\n\n"
        "Tweets:\n" + context +
        "\nQuestion: " + question +
        "\nAnswer:"
    )
    output = generator(prompt)
    return output[0]["generated_text"].strip()


# ── UI ────────────────────────────────────────────────────────────────────────

# Header
st.markdown("## 🐦 Tweet Sentiment Analyser")
st.caption("Find out how people feel about any topic using 50,000 real tweets")
st.divider()

# Load models
with st.spinner("Loading models... this takes 1-2 mins the first time ☕"):
    index, metadata, embedder, generator = load_all_models()

# Keep selected question in memory
if "selected" not in st.session_state:
    st.session_state.selected = ""

# Example questions section
st.markdown("**💡 Click a topic to explore:**")

# Row 1 of buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("☕ Coffee", use_container_width=True):
        st.session_state.selected = "What do people think about coffee?"
with col2:
    if st.button("😴 Mondays", use_container_width=True):
        st.session_state.selected = "How do people feel about Mondays?"
with col3:
    if st.button("🎂 Birthdays", use_container_width=True):
        st.session_state.selected = "How do people feel about birthdays?"

# Row 2 of buttons
col4, col5, col6 = st.columns(3)
with col4:
    if st.button("🎵 Music", use_container_width=True):
        st.session_state.selected = "What are positive tweets about music?"
with col5:
    if st.button("☀️ Summer", use_container_width=True):
        st.session_state.selected = "Do people like summer?"
with col6:
    if st.button("🍕 Pizza", use_container_width=True):
        st.session_state.selected = "Do people love pizza?"

st.divider()

# Search box
st.markdown("**🔍 Or type your own question:**")
question = st.text_input(
    "",
    value=st.session_state.selected,
    placeholder="e.g. how do people feel about coffee?",
    label_visibility="collapsed"
)

search = st.button("Analyse →", type="primary", use_container_width=True)

# Results
if search and question:
    with st.spinner("Searching through 50,000 tweets..."):
        tweets = find_tweets(question, index, metadata, embedder)
        answer = generate_answer(question, tweets, generator)

    pos_tweets = [t for t in tweets if t["sentiment"] == "positive"]
    neg_tweets = [t for t in tweets if t["sentiment"] == "negative"]
    total      = len(tweets)
    pos_pct    = round(len(pos_tweets) / total * 100) if total > 0 else 0
    neg_pct    = 100 - pos_pct

    st.divider()

    # Sentiment score bar
    st.markdown("**📊 Sentiment Overview**")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.metric("😊 Positive", f"{pos_pct}%", f"{len(pos_tweets)} tweets")
    with col_neg:
        st.metric("😔 Negative", f"{neg_pct}%", f"{len(neg_tweets)} tweets")

    st.progress(pos_pct / 100)

    # AI Answer
    st.markdown("**🤖 AI Analysis**")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Tweets in tabs
    st.markdown("**📄 Tweets Found**")
    tab1, tab2, tab3 = st.tabs([
        f"All ({total})",
        f"😊 Positive ({len(pos_tweets)})",
        f"😔 Negative ({len(neg_tweets)})"
    ])

    with tab1:
        for t in tweets:
            css  = "tweet-positive" if t["sentiment"] == "positive" else "tweet-negative"
            icon = "😊" if t["sentiment"] == "positive" else "😔"
            st.markdown(f'<div class="{css}">{icon} {t["clean_text"]}</div>', unsafe_allow_html=True)

    with tab2:
        if pos_tweets:
            for t in pos_tweets:
                st.markdown(f'<div class="tweet-positive">😊 {t["clean_text"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No positive tweets found.")

    with tab3:
        if neg_tweets:
            for t in neg_tweets:
                st.markdown(f'<div class="tweet-negative">😔 {t["clean_text"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No negative tweets found.")

elif not question:
    st.info("👆 Click a topic above or type your own question to get started!")
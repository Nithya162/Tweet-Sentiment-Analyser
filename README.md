# 🐦 Tweet Sentiment Analyser

> **Status: 🚧 Ongoing Project**

A RAG-based chatbot that lets you explore how people feel about any topic using **1.6 million real tweets** from the Sentiment140 dataset. Ask a question, get an AI-generated answer backed by real tweet evidence.

---

## 💬 Example

> **You ask:** What do people think about coffee?
>
> **Bot:** People are very positive about coffee, especially in the morning!
>
> 😊 *i love my morning coffee so much*
> 😊 *coffee makes everything better*
> 😔 *coffee gives me headaches*

---

## 🏗️ How It Works

```
1.6M Tweets (Sentiment140)
        │
        ▼
┌─────────────────────┐
│   load_data.py      │  Load CSV → Clean tweets → Save 50K sample
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   embed_index.py    │  Convert tweets → numbers → Build FAISS index
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   chatbot.py        │  Search tweets → Generate answer
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   app.py            │  Streamlit web interface
└─────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Dataset | Sentiment140 — 1.6M tweets (Kaggle) |
| ETL & Cleaning | Python, Pandas, Regex |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| LLM | google/flan-t5-base (free, runs on CPU) |
| Interface | Streamlit |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Nithya162/tweet-sentiment-analyser.git
cd tweet-sentiment-analyser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/kazanova/sentiment140
- Download and place at: `data/raw/dataset.csv`

### 4. Run in order
```bash
python load_data.py
python embed_index.py
streamlit run app.py
```

---

## 📁 Project Structure

```
tweet-sentiment-analyser/
├── data/
│   ├── raw/        ← place dataset.csv here
│   ├── clean/      ← auto-generated
│   └── index/      ← auto-generated
├── load_data.py    ← cleans and samples 50K tweets
├── embed_index.py  ← builds FAISS search index
├── chatbot.py      ← terminal chatbot
├── app.py          ← Streamlit web interface
├── requirements.txt
└── README.md
```

---

## 🚧 Ongoing Work

- [ ] Deploy on Streamlit Cloud
- [ ] Add date-based filtering
- [ ] Add word cloud per topic
- [ ] Upgrade to a larger LLM for better answers

---

## 👩‍💻 Author

**NithyaShree RaviKumar**
M.S. Data Science, Analytics and Engineering — Arizona State University
[LinkedIn](https://www.linkedin.com/in/nithya-shree-ravi-kumar-b3b010241) | [GitHub](https://github.com/Nithya162) | [Scholar](https://scholar.google.com/citations?user=KtkZ8fgAAAAJ&hl=en)

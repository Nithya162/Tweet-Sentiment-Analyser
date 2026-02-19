# 🐦 Tweet Sentiment Analyser

> **Status: 🚧 Ongoing Project**

A Retrieval-Augmented Generation (RAG) chatbot that lets you explore how people feel about any topic using **1.6 million real tweets** from the Sentiment140 dataset. Ask a question, get an AI-generated answer backed by real tweet evidence.

---

## 📸 Demo

> *Ask: "What do people think about coffee?"*
> 
> **Bot:** People are overwhelmingly positive about coffee, especially in the morning. A small number of users mention negative experiences like headaches.
> 
> 😊 *i love my morning coffee so much*  
> 😊 *coffee makes everything better*  
> 😔 *coffee gives me headaches i hate it*

---

## 🏗️ How It Works

```
1.6M Tweets (Sentiment140)
        │
        ▼
┌─────────────────────┐
│  ETL Pipeline       │  Load → Clean → Sample 50K tweets
│  step1_load_data.py │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Embedding + Index  │  Convert text → numbers → FAISS index
│  step2_embed_index  │  (sentence-transformers/all-MiniLM-L6-v2)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  RAG Chatbot        │  Search → Retrieve → Generate answer
│  step3_chatbot.py   │  (google/flan-t5-base)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Streamlit UI       │  Interactive web interface
│  app.py             │  Clickable topics + sentiment tabs
└─────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Dataset | Sentiment140 (1.6M tweets, Kaggle) |
| ETL & Cleaning | Python, Pandas, Regex |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| LLM | `google/flan-t5-base` (free, runs on CPU) |
| RAG Pipeline | Custom built with HuggingFace Transformers |
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

### 4. Run the pipeline in order
```bash
# Step 1 - clean the data
python step1_load_data.py

# Step 2 - build the search index (takes ~5 mins)
python step2_embed_index.py

# Step 3 - launch the app
streamlit run app.py
```

---

## 💬 Example Questions

- *What do people think about coffee?*
- *Show me negative tweets about Mondays*
- *How do people feel about birthdays?*
- *Do people like summer?*
- *What are positive tweets about music?*

---

## 📁 Project Structure

```
tweet-sentiment-analyser/
├── data/
│   ├── raw/          ← place dataset.csv here
│   ├── clean/        ← auto-generated after step 1
│   └── index/        ← auto-generated after step 2
├── step1_load_data.py    ← ETL: load, clean, sample tweets
├── step2_embed_index.py  ← embed tweets + build FAISS index
├── step3_chatbot.py      ← terminal chatbot (optional)
├── app.py                ← Streamlit web interface
├── requirements.txt
└── README.md
```

---

## 🔧 Requirements

```
pandas
numpy
faiss-cpu
sentence-transformers==2.7.0
transformers
torch==2.1.0
torchvision==0.16.0
streamlit
tqdm
```

---

## 🚧 Ongoing Work

- [ ] Add neutral sentiment class
- [ ] Deploy on Streamlit Cloud (public URL)
- [ ] Add date-based filtering (see how sentiment changes over time)
- [ ] Swap flan-t5 for a larger LLM for better answers
- [ ] Add word cloud visualisation per topic

---

## 👩‍💻 Author

**NithyaShree RaviKumar**  
M.S. Data Science, Analytics and Engineering — Arizona State University  
[LinkedIn](https://www.linkedin.com/in/nithya-shree-ravi-kumar-b3b010241) | [GitHub](https://github.com/Nithya162) | [Scholar](https://scholar.google.com/citations?user=KtkZ8fgAAAAJ&hl=en)

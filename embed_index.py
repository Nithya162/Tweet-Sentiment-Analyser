import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Your file locations
CLEAN_PATH = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\clean\clean_tweets.csv"
INDEX_PATH = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\faiss.index"
META_PATH  = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\metadata.pkl"


# STEP 1 - Load the clean CSV we created in step 1
print("Loading clean tweets...")
df = pd.read_csv(CLEAN_PATH)
print("Total tweets loaded:", len(df))
print("Sentiment counts:")
print(df["sentiment"].value_counts())


# STEP 2 - Load the embedding model
print("")
print("Loading embedding model")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded!")


# STEP 3 - Convert tweets into numbers (embeddings)
print("")
print("Converting tweets into numbers... this takes 5-10 mins")

texts = df["embed_text"].tolist()
all_embeddings = []

for i in tqdm(range(0, len(texts), 256), desc="Progress"):
    batch = texts[i : i + 256]
    embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    all_embeddings.append(embeddings)

matrix = np.vstack(all_embeddings).astype("float32")
print("Done! Shape:", matrix.shape)


# STEP 4 - Build the FAISS search index
print("")
print("Building search index...")
dim   = matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(matrix)
print("Index ready! Total tweets indexed:", index.ntotal)


# STEP 5 - Save the index and metadata to disk
print("")
print("Saving files...")

Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, INDEX_PATH)
print("Saved index to:", INDEX_PATH)

metadata = df[["sentiment", "clean_text", "user", "date"]].to_dict(orient="records")
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)
print("Saved metadata to:", META_PATH)


# STEP 6 - Quick test to make sure search works
print("")
print("Testing search...")

test_queries = [
    "positive coffee morning",
    "negative monday work",
    "positive happy birthday",
]

for query in test_queries:
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=3)
    print("")
    print("Query:", query)
    for score, idx in zip(scores[0], indices[0]):
        row = metadata[idx]
        print(" -", row["sentiment"].upper(), "| score:", round(score, 3), "|", row["clean_text"][:80])
print("")

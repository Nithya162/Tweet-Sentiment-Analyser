import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Your file locations
INDEX_PATH = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\faiss.index"
META_PATH  = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\index\metadata.pkl"


# STEP 1 - Load the FAISS index and metadata we saved in step 2
print("Loading search index...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Index loaded! Total tweets:", index.ntotal)


# STEP 2 - Load the embedding model (same one we used in step 2)
print("")
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model ready!")


# STEP 3 - Load the LLM (this is what generates the actual answer)
print("")
print("Loading the answer generator model... this takes 1-2 mins first time")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

answer_generator = pipeline(
    "text2text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    temperature=None,
    top_p=None,
)
print("Answer generator ready!")


# STEP 4 - Function to search for similar tweets
def find_similar_tweets(question):
    # convert question to numbers
    question_numbers = embedder.encode([question], normalize_embeddings=True).astype("float32")

    # search FAISS for top 5 most similar tweets
    scores, indices = index.search(question_numbers, k=5)

    # get the actual tweet text using the indices
    results = []
    for score, idx in zip(scores[0], indices[0]):
        tweet = metadata[idx]
        results.append(tweet)

    return results


# STEP 5 - Function to generate an answer using the tweets found
def generate_answer(question, tweets, history):
    # combine all found tweets into one block of text
    context = ""
    for tweet in tweets:
        context += tweet["sentiment"].upper() + ": " + tweet["clean_text"] + "\n"

    # add conversation history so the bot remembers previous questions
    history_text = ""
    if history:
        for old_question, old_answer in history[-4:]:
            history_text += "User: " + old_question + "\n"
            history_text += "Bot: " + old_answer + "\n"

    # build the prompt (instructions for the AI)
    prompt = """You are a helpful assistant that analyzes tweet sentiments.
Use the tweets below to answer the question.

""" + history_text + """
Tweets:
""" + context + """
Question: """ + question + """
Answer:"""

    # generate the answer
    output = answer_generator(prompt)
    answer = output[0]["generated_text"].strip()
    return answer


# STEP 6 - Chat loop (keeps the conversation going)
print("")
print("Chatbot is ready!")
print("You can ask things like:")
print("  - what do people think about coffee?")
print("  - show me negative tweets about mondays")
print("  - how do people feel about birthdays?")
print("")
print("Type 'quit' to stop")
print("")

history = []   # this stores your conversation so the bot remembers

while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() == "quit":
        print("Bye!")
        break

    # find similar tweets
    tweets = find_similar_tweets(question)

    # generate answer
    answer = generate_answer(question, tweets, history)

    # show answer
    print("")
    print("Bot:", answer)
    print("")

    # save to history so bot remembers
    history.append((question, answer))
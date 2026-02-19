import pandas as pd
import re
from pathlib import Path

# file locations
RAW_PATH   = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\raw\dataset.csv"
CLEAN_PATH = r"C:\Users\nithy\OneDrive\Desktop\Sentiment analysis with tweets\data\clean\clean_tweets.csv"


# STEP 1 - Load the CSV file
print("Loading the dataset")

df = pd.read_csv(
    RAW_PATH,
    encoding="latin-1",
    header=None,
    names=["target", "id", "date", "flag", "user", "text"]
)

print("Done loading!")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
print("")
print("First 3 rows:")
print(df[["target", "user", "text"]].head(3))


# STEP 2 - Look at the data
print("")
print("Negative tweets (target=0):", (df["target"] == 0).sum())
print("Positive tweets (target=4):", (df["target"] == 4).sum())
print("")
print("Example negative tweet:", df[df["target"] == 0]["text"].iloc[0])
print("Example positive tweet:", df[df["target"] == 4]["text"].iloc[0])


# STEP 3 - Take a sample (50,000 tweets out of 1.6 million)
print("")
print("Taking a sample of 50,000 tweets...")

neg = df[df["target"] == 0].sample(n=25000, random_state=42)
pos = df[df["target"] == 4].sample(n=25000, random_state=42)

small = pd.concat([neg, pos]).reset_index(drop=True)
print("Sample ready:", len(small), "tweets")


# STEP 4 - Add a readable label (positive or negative)
def get_label(number):
    if number == 4:
        return "positive"
    else:
        return "negative"

small["sentiment"] = small["target"].apply(get_label)


# STEP 5 - Clean the tweet text
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)       # remove links
    text = re.sub(r'@\w+', '', text)           # remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)      # remove # symbol
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove emojis
    text = re.sub(r'\s+', ' ', text)           # fix extra spaces
    return text.strip().lower()

print("Cleaning tweets...")
small["clean_text"] = small["text"].apply(clean_tweet)


# STEP 6 - Remove tweets that are too short after cleaning
small = small[small["clean_text"].str.len() > 15]
small = small.drop_duplicates(subset="clean_text")
small = small.reset_index(drop=True)

print("Clean tweets ready:", len(small))


# STEP 7 - Create embed_text column (sentiment + tweet together)
small["embed_text"] = small["sentiment"] + " " + small["clean_text"]


# STEP 8 - Keep only the columns we need
final = small[["sentiment", "clean_text", "embed_text", "user", "date"]]

print("")
print("Sample of clean data:")
print(final[["sentiment", "clean_text"]].head(5))


# STEP 9 - Save the clean file
Path(CLEAN_PATH).parent.mkdir(parents=True, exist_ok=True)
final.to_csv(CLEAN_PATH, index=False)

print("")
print("Saved clean file to:", CLEAN_PATH)
print("")
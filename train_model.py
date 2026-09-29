# 1 IMPORT LIBRARIES
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#2.DATASET
data = {
    "text": [
        "I love this product",
        "This product is excellent",
        "Amazing service and friendly staff",
        "The experience was wonderful",
        "I am very happy with the purchase",
        "The quality is fantastic",
        "This is a great application",
        "I really enjoyed the movie",
        "The support team was helpful",
        "Very good experience",

        "I hate this product",
        "This product is terrible",
        "The service was horrible",
        "I am very disappointed",
        "The quality is poor",
        "This is a bad experience",
        "I did not like the movie",
        "The support was useless",
        "Very frustrating experience",
        "I regret buying this product",

        "The package arrived today",
        "The product is black",
        "The meeting starts at ten",
        "I received the order yesterday",
        "The laptop has 16 GB RAM",
        "The store is open today",
        "The course has four modules",
        "The phone was delivered in the morning",
        "The application was updated",
        "The class starts at nine"
    ],

    "sentiment": (
        ["positive"] * 10 +
        ["negative"] * 10 +
        ["neutral"] * 10
    )
}

df = pd.DataFrame(data)

print(df.head())

# 3.TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print(df[["text", "clean_text"]].head())

#4. Separate text and sentiment
X = df["clean_text"]
y = df["sentiment"]

#5.CONVERT TEXT INTO NUMBERS USING TF-IDF VECTORIZER
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF matrix shape:", X_tfidf.shape)

# 6 — Train Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_tfidf, y)

print("Logistic Regression model trained successfully.")

joblib.dump(model, "model.pkl")   #joblib.dump() saves a Python object into a file Why .pkl?
 #.pkl is a pickle file. It allows us to save a Python object and load it again later.
# So:
# model
#   ↓
# joblib.dump()
#   ↓
# model.pkl

joblib.dump(vectorizer, "vectorizer.pkl") #"Save my trained TF-IDF vectorizer into a file called vectorizer.pkl."

# So now you have:
# vectorizer
#      ↓
# joblib.dump()
#      ↓
# vectorizer.pkl

print("Model saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
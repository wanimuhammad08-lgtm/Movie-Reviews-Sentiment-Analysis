# Movie Reviews Sentiment Analysis (Final Version)

# ----------------------------
# Import Libraries
# ----------------------------
import numpy as np
import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Load Dataset
# ----------------------------
print("Loading dataset...")

dataset = pd.read_csv("Dataset/IMDB.csv")

print("Dataset Shape:", dataset.shape)


# ----------------------------
# Encode Labels
# ----------------------------
dataset["sentiment"] = dataset["sentiment"].replace({
    "positive": 1,
    "negative": 0
}).astype(int)


# ----------------------------
# Text Cleaning (Minimal )
# ----------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = text.lower()                # lowercase
    return text


print("\nCleaning text...")
dataset["review"] = dataset["review"].apply(clean_text)

print("\nSample review:")
print(dataset["review"].iloc[0])


# ----------------------------
# TF-IDF Feature Extraction 
# ----------------------------
print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=20000,      
    ngram_range=(1, 2),     
    stop_words=None          
)

X = tfidf.fit_transform(dataset["review"])
y = dataset["sentiment"].values

print("X Shape:", X.shape)


# ----------------------------
# Train-Test Split
# ----------------------------
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)


# ----------------------------
# Train Multiple Models 
# ----------------------------
print("\nTraining models...\n")

models = {
    "Multinomial NB": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(C=2.0),   # tuned
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

    print(f"{name} Accuracy: {acc * 100:.2f}%\n")


# ----------------------------
# Model Comparison
# ----------------------------
print("\nFinal Model Comparison:\n")

for name, acc in results.items():
    print(f"{name}: {acc * 100:.2f}%")

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print(f"\nBest Model: {best_model_name} ")


# ----------------------------
# Detailed Report (Best Model)
# ----------------------------
y_pred_best = best_model.predict(X_test)

print("\nClassification Report (Best Model):\n")
print(classification_report(y_test, y_pred_best))


# ----------------------------
# Save Best Model
# ----------------------------
print("\nSaving best model...")

joblib.dump(best_model, "Models/best_model.pkl")
joblib.dump(tfidf, "Models/tfidf_vectorizer.pkl")


print("\nTraining completed successfully! ")
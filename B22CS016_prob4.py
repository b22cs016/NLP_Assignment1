import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# 1. DATA COLLECTION
# We use the 20 Newsgroups dataset, specifically 'rec.sport.hockey', 'rec.sport.baseball'
# and 'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'
print("Loading dataset...")
categories = [
    'rec.sport.hockey', 'rec.sport.baseball',  # SPORTS
    'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast' # POLITICS
]

# Fetch data
dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Create labels: 0 for Sports, 1 for Politics
# rec.sport.* indices are 0 and 1 in our list selection? No, let's map manually.
y_binary = []
for label in dataset.target:
    # dataset.target_names gives the actual string names
    name = dataset.target_names[label]
    if 'sport' in name:
        y_binary.append('Sports')
    else:
        y_binary.append('Politics')

X = dataset.data
y = np.array(y_binary)

print(f"Data Loaded: {len(X)} documents.")
print(f"Sports examples: {np.sum(y == 'Sports')}")
print(f"Politics examples: {np.sum(y == 'Politics')}")

# 2. PREPROCESSING & SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. MODEL TRAINING & COMPARISON
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(dual='auto', max_iter=1000)
}

results = {}

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# 4. VISUALIZATION (Optional, saves a plot for your report)
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy Score")
plt.title("Model Comparison: Sports vs. Politics")
plt.savefig("model_comparison.png")
print("\nPlot saved as 'model_comparison.png'.")
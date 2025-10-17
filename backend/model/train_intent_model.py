import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ==============================
# 1. Load the dataset
# ==============================
print("📂 Loading dataset...")
data = pd.read_csv("../data/faq_dataset.csv", encoding='utf-8-sig')

# Check basic info
print(f"✅ Loaded {len(data)} samples with {data['intent'].nunique()} unique intents.\n")

# ==============================
# 2. Split data for training/testing
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    data['query'],
    data['intent'],
    test_size=0.2,
    random_state=42,
    stratify=data['intent']  # ensures balanced split per intent
)

# ==============================
# 3. Text to numeric features (TF-IDF)
# ==============================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),     # use unigrams and bigrams
    max_features=3000       # limit features to prevent overfitting
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 4. Train the model
# ==============================
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_vec, y_train)

# ==============================
# 5. Evaluate the model
# ==============================
acc = model.score(X_test_vec, y_test)
print(f"\n🎯 Model Accuracy: {acc:.2f}\n")

# Detailed classification report
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: show confusion matrix
print("📈 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 6. Test with sample queries
# ==============================
sample_queries = [
    "I forgot my password",
    "My card was stolen",
    "I want to close my account",
    "Where is my refund?",
    "How can I contact support?",
    "My payment failed",
    "What is the interest rate on my credit card?"
]

print("\n🧠 Sample Predictions:")
for query in sample_queries:
    intent = model.predict(vectorizer.transform([query]))[0]
    print(f" - {query} → {intent}")

# ==============================
# 7. Save trained model and vectorizer
# ==============================
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully as 'intent_model.pkl' and 'vectorizer.pkl'.")

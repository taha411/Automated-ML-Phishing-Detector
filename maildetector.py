# =====================================================================
# PHASE 1: DATA PREPARATION & TOOLKIT INITIALIZATION
# =====================================================================
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download the required NLTK resource packages (including environment-specific updates)
nltk.download('punkt')
nltk.download('punkt_tab')                 # Fixed missing tab package for newer Python versions
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Step 1: Read the dataset & format it
df = pd.read_csv('messages.csv')

# Handle missing data matching presentation logic (Do NOT drop rows)
df['subject'] = df['subject'].fillna('') 

# Combine subject and message text for comprehensive NLP feature reading
df['clean_text'] = df['subject'] + " " + df['message']

# Step 2: Encode labels (0: Legitimate/Benign, 1: Spam/Phishing/Malicious)
# Our messages.csv already has numerical 0 and 1 labels applied

# Print the 83/17 Baseline Class Distribution Insight
total_emails = len(df)
legit_count = len(df[df['label'] == 0])
spam_count = len(df[df['label'] == 1])
print("--- Class Distribution Baseline ---")
print(f"Legitimate Emails (0): {round((legit_count / total_emails) * 100, 2)}%")
print(f"Spam/Phishing Emails (1): {round((spam_count / total_emails) * 100, 2)}%\n")


# =====================================================================
# PHASE 2: TEXT PRE-PROCESSING (NLP Pipeline Function)
# =====================================================================
def preprocess_text(text):
    # FORCE DATA TO STRING TYPE (Fixes the float/NaN error from empty fields)
    text = str(text) 
    
    # Step 3: Convert text to lowercase
    text = text.lower()
    
    # Step 4: Remove punctuation & Tokenize using Punkt
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalnum()]
    
    # Step 5: Remove stop words
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in words if word not in stop_words]
    
    # Re-join tokens back into a single clean string sentence
    return " ".join(cleaned_tokens)

# Apply the pre-processing pipeline across the text column
df['clean_text'] = df['clean_text'].apply(preprocess_text)


# =====================================================================
# PHASE 3: MODELING (Applying the Correct Professional Fix)
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df['clean_text']  # Our processed text features
y = df['label']       # Our encoded labels

# --- THE FIX ---
# Step 8 executed BEFORE Step 6: Split raw text into isolated train/test groups
# This keeps the test data 100% secret to avoid Data Leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 6: Feature Extraction (Convert text into vectors using only training vocabulary)
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train) # Learns dictionary from training group
X_test_vectors = vectorizer.transform(X_test)       # Uses strict existing training dictionary

# Step 7: Import and initialize the classifier
classifier = MultinomialNB()

# Execute model training phase
classifier.fit(X_train_vectors, y_train)


# =====================================================================
# PHASE 4: EVALUATION
# =====================================================================
# Step 9: Make predictions and check accuracy performance on the hidden test set
y_predictions = classifier.predict(X_test_vectors)

print("--- Model Evaluation Metrics ---")
print(f"Final Test Accuracy Score: {accuracy_score(y_test, y_predictions) * 100:.2f}%")
print("\nConfusion Matrix Grid:")
print(confusion_matrix(y_test, y_predictions))
print("\nClassification Precision & Recall Report:")
print(classification_report(y_test, y_predictions))
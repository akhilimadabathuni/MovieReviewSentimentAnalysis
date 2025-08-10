# main.py

# ---------------------------------------------------
# Step 1: Import Necessary Libraries
# ---------------------------------------------------
# Pandas for data manipulation
import pandas as pd
# Seaborn and Matplotlib for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Regular expressions for text cleaning
import re
# NLTK for text processing (stopwords and lemmatization)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Scikit-learn for machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Download NLTK data (only need to run once) ---
nltk.download('stopwords')
nltk.download('wordnet')

print("Libraries imported successfully.")


# ---------------------------------------------------
# Step 2: Load and Explore the Data
# ---------------------------------------------------
# For this starter, we'll create a small, representative dataset.
# In a real project, you would load this from a CSV file, e.g., pd.read_csv('imdb_reviews.csv')
# This new line reads the data from your CSV file
df = pd.read_csv('IMDB Dataset.csv')
print("\n--- Data Head ---")
print(df.head())

print("\n--- Data Info ---")
df.info()

# --- Visualize the sentiment distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.show()


# ---------------------------------------------------
# Step 3: Text Preprocessing
# ---------------------------------------------------
# We need to clean the text to make it suitable for the model.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase the text
    text = text.lower()
    # 2. Remove HTML tags and URLs
    text = re.sub(r'<.*?>|https?://\S+|www\.\S+', '', text)
    # 3. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # 4. Tokenize the text (split into words)
    tokens = text.split()
    # 5. Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 6. Join tokens back into a string
    return " ".join(cleaned_tokens)

# Apply the preprocessing function to the 'review' column
df['cleaned_review'] = df['review'].apply(preprocess_text)

print("\n--- Data after Preprocessing ---")
print(df[['review', 'cleaned_review']].head())


# ---------------------------------------------------
# Step 4: Feature Extraction (Vectorization)
# ---------------------------------------------------
# Convert the cleaned text into numerical vectors using TF-IDF.
vectorizer = TfidfVectorizer(max_features=1000) # Use top 1000 words

X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# --- Split data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# ---------------------------------------------------
# Step 5: Build and Train the Model
# ---------------------------------------------------
# We'll use Logistic Regression, a simple and effective model for text classification.
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# ---------------------------------------------------
# Step 6: Evaluate the Model
# ---------------------------------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# --- Calculate Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# --- Classification Report ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ---------------------------------------------------
# Step 7: Test the Model with Your Own Reviews!
# ---------------------------------------------------
def predict_sentiment(new_review):
    # Preprocess the new review
    cleaned_review = preprocess_text(new_review)
    # Vectorize the cleaned review
    vectorized_review = vectorizer.transform([cleaned_review])
    # Predict the sentiment
    prediction = model.predict(vectorized_review)
    probability = model.predict_proba(vectorized_review)
    
    # Get the confidence score
    confidence = max(probability[0])
    
    return prediction[0], confidence

# --- Interactive Loop ---
print("\n--- Live Sentiment Analyzer ---")
print("Type a movie review and press Enter. (Type 'quit' to exit)")

while True:
    user_input = input("\nEnter your review: ")
    if user_input.lower() == 'quit':
        break
    
    prediction, confidence = predict_sentiment(user_input)
    print(f"  -> Predicted Sentiment: {prediction.upper()}")
    print(f"  -> Confidence: {confidence:.2f}")

# --- Example 1: Positive Review ---
review1 = "This movie was absolutely fantastic! The visuals were stunning."
print(f"\nReview: '{review1}'")
print(f"Predicted Sentiment: {predict_sentiment(review1)}")

# --- Example 2: Negative Review ---
review2 = "A complete waste of money and time. The plot made no sense."
print(f"\nReview: '{review2}'")
print(f"Predicted Sentiment: {predict_sentiment(review2)}")
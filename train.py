import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    """
    Trains a machine learning model to classify strings as safe or malicious.
    """
    print("Starting model training...")

    # 1. Create our training dataset
    # In a real-world scenario, you'd use thousands of examples from public datasets.
    # 0 = safe, 1 = malicious
    data = [
        ("google.com", 0),
        ("127.0.0.1", 0),
        ("localhost", 0),
        ("my-pc", 0),
        ("a-valid-hostname", 0),
        ("github.com", 0),
        ("8.8.8.8 && ls", 1),
        ("google.com; whoami", 1),
        ("; rm -rf /", 1),
        ("cat /etc/passwd", 1),
        ("$(uname -a)", 1),
        ("`id`", 1),
        ("some-host | id", 1),
    ]

    # Separate the text from the labels (0 or 1)
    texts, labels = zip(*data)

    # 2. Vectorize the text data
    # This converts text strings into a numerical format (TF-IDF features).
    # 'char_wb' analyzes character n-grams, which is effective for finding
    # suspicious patterns in command injection strings.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    X = vectorizer.fit_transform(texts)

    # 3. Train the classification model
    # Logistic Regression is a simple, fast, and effective model for this task.
    model = LogisticRegression()
    model.fit(X, labels)

    # 4. Save the trained vectorizer and model to files
    # These files will be loaded by our main application.
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model, 'command_injection_model.pkl')

    print("✅ Model training complete!")
    print("   - Saved vectorizer to 'tfidf_vectorizer.pkl'")
    print("   - Saved model to 'command_injection_model.pkl'")

if __name__ == "__main__":
    train_model()
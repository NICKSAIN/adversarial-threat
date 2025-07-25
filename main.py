# main_ai.py
import joblib

# --- LOAD THE TRAINED MODEL AND VECTORIZER ---
# These files are created during your offline training phase.
try:
    model = joblib.load('command_injection_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please train the model first.")
    exit()


def detect_with_ai(input_string):
    """
    Detects potential command injection using a trained ML model.

    Args:
        input_string: The user-provided string to analyze.

    Returns:
        A boolean: True if an attack is predicted, False otherwise.
    """
    # 1. Convert the new input string into the same numerical format
    #    as the training data. Note it must be in a list or iterable.
    input_vector = vectorizer.transform([input_string])

    # 2. Use the loaded model to predict if it's malicious (1) or safe (0)
    prediction = model.predict(input_vector)

    # 3. Return the result
    return prediction[0] == 1 # Returns True if prediction is 1

# The rest of your main() function would call this new detector
def main():
    print("--- AI-Powered Command Injection Detector ---")
    # ... (rest of the main loop is the same)

    while True:
        user_input = input("Enter hostname to ping > ")
        # ... (exit logic)

        if detect_with_ai(user_input):
            print("🚨 ALERT: Potential Attack Detected by AI! 🚨\n")
        else:
            print("✅ Input appears safe.\n")

# ... (if __name__ == "__main__": etc.)
import joblib
import os
import sys

# -----------------------------------
# Load model and vectorizer
# -----------------------------------
def load_model_and_vectorizer():
    model_path = 'command_injection_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("🚨 Error: Model or vectorizer file not found!")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# -----------------------------------
# AI Prediction Function
# -----------------------------------
def detect_with_ai(input_string, model, vectorizer):
    input_vector = vectorizer.transform([input_string])
    prediction = model.predict(input_vector)
    return prediction[0] == 1

# -----------------------------------
# Simulate Adversarial Modification
# -----------------------------------
def generate_adversarial_input(input_string):
    return input_string.replace(";", "%3B").replace("&", "&&").replace("|", "||").replace("$", "\\$")

# -----------------------------------
# Test Evasion Attacks
# -----------------------------------
def test_evasion_samples(model, vectorizer):
    print("\n--- Testing Known Evasion Techniques ---")
    evasion_samples = [
        "google.com; ls",
        "127.0.0.1 && whoami",
        "`ping google.com`",
        "localhost | cat /etc/passwd",
        "bing.com$(reboot)",
        "8.8.8.8 || echo hacked",
        "$(shutdown now)",
        "myhost.com; rm -rf /"
    ]

    for sample in evasion_samples:
        adv_sample = generate_adversarial_input(sample)
        detected = detect_with_ai(adv_sample, model, vectorizer)
        status = "🚨 DETECTED" if detected else "❌ MISSED"
        print(f"{status}: {adv_sample}")

# -----------------------------------
# Interactive Mode
# -----------------------------------
def main():
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return

    print("\n--- AI-Powered Command Injection Detector ---")
    print("Type a hostname or command. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter hostname to ping > ")

        if user_input.lower() == 'exit':
            print("Exiting. Stay safe!")
            break

        print("-" * 30)
        if detect_with_ai(user_input, model, vectorizer):
            print("🚨 ALERT: Potential Attack Detected by AI!")
        else:
            print("✅ Input appears safe.")

        # Try adversarial version
        adv_input = generate_adversarial_input(user_input)
        if adv_input != user_input:
            print(f"\nTesting adversarial form: {adv_input}")
            if detect_with_ai(adv_input, model, vectorizer):
                print("⚠️ Detected adversarial attempt!")
            else:
                print("❌ Model failed to detect adversarial form.")

        print("-" * 30 + "\n")

# -----------------------------------
# Entry Point
# -----------------------------------
if __name__ == "__main__":
    if "--test-attacks" in sys.argv:
        model, vectorizer = load_model_and_vectorizer()
        if model:
            test_evasion_samples(model, vectorizer)
    else:
        main()

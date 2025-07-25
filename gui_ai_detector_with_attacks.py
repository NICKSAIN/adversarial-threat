import joblib
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
def load_model_and_vectorizer():
    try:
        model = joblib.load("command_injection_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        messagebox.showerror("Error", "Model or vectorizer not found.")
        return None, None

# -------------------------------
# Detect Command Injection
# -------------------------------
def detect_with_ai(input_string, model, vectorizer):
    input_vector = vectorizer.transform([input_string])
    prediction = model.predict(input_vector)
    return prediction[0] == 1

# -------------------------------
# Adversarial Input Generator
# -------------------------------
def generate_adversarial_input(input_string):
    return input_string.replace(";", "%3B").replace("&", "&&").replace("|", "||").replace("$", "\\$")

# -------------------------------
# Retrain Model
# -------------------------------
def train_new_model_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        X = df['text']
        y = df['label']

        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_vec, y)

        joblib.dump(model, "command_injection_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

        return True
    except Exception as e:
        messagebox.showerror("Training Error", str(e))
        return False

# -------------------------------
# GUI Application
# -------------------------------
class CommandInjectionGUI:
    def __init__(self, root):
        self.model, self.vectorizer = load_model_and_vectorizer()
        self.root = root
        self.root.title("AI Command Injection Detector")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self.attack_samples = [
            "google.com; ls",
            "127.0.0.1 && whoami",
            "`ping google.com`",
            "localhost | cat /etc/passwd",
            "bing.com$(reboot)",
            "8.8.8.8 || echo hacked",
            "$(shutdown now)",
            "myhost.com; rm -rf /",
            "curl http://attacker.site",
            "google.com; sleep 10"
        ]

        # Input Section
        self.label = tk.Label(root, text="Enter hostname or input string:", font=("Arial", 12))
        self.label.pack(pady=10)

        self.input_entry = tk.Entry(root, width=50, font=("Arial", 12))
        self.input_entry.pack(pady=5)

        self.detect_button = tk.Button(root, text="Detect Injection", command=self.run_detection,
                                       bg="#4CAF50", fg="white", font=("Arial", 12))
        self.detect_button.pack(pady=10)

        self.output_text = tk.Text(root, height=10, width=70, font=("Arial", 10))
        self.output_text.pack(pady=10)

        # Training Section
        self.train_button = tk.Button(root, text="Train New Model from CSV", command=self.load_csv_and_train,
                                      bg="#007BFF", fg="white", font=("Arial", 11))
        self.train_button.pack(pady=5)

        # Attack Test Section
        self.attack_test_button = tk.Button(root, text="Test Model with Attacks", command=self.test_attack_samples,
                                            bg="#DC3545", fg="white", font=("Arial", 11))
        self.attack_test_button.pack(pady=5)

    def run_detection(self):
        if not self.model or not self.vectorizer:
            return

        user_input = self.input_entry.get().strip()
        if not user_input:
            messagebox.showwarning("Input Needed", "Please enter a string.")
            return

        self.output_text.delete("1.0", tk.END)

        is_attack = detect_with_ai(user_input, self.model, self.vectorizer)
        result = "🚨 ALERT: Potential Attack Detected!" if is_attack else "✅ Input appears safe."
        self.output_text.insert(tk.END, f"Original Input:\n{result}\n\n")

        adv_input = generate_adversarial_input(user_input)
        if adv_input != user_input:
            is_adv_attack = detect_with_ai(adv_input, self.model, self.vectorizer)
            adv_result = "⚠️ Detected in adversarial form!" if is_adv_attack else "❌ Bypassed detection in adversarial form."
            self.output_text.insert(tk.END, f"Adversarial Variant:\n{adv_result}\n")

    def load_csv_and_train(self):
        filepath = filedialog.askopenfilename(title="Select CSV Training File", filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

        success = train_new_model_from_csv(filepath)
        if success:
            self.model, self.vectorizer = load_model_and_vectorizer()
            messagebox.showinfo("Training Complete", "Model retrained and loaded successfully!")

    def test_attack_samples(self):
        if not self.model or not self.vectorizer:
            messagebox.showwarning("Model Missing", "Please load or train a model first.")
            return

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "🔍 Testing known attack inputs:\n\n")

        for attack in self.attack_samples:
            result = detect_with_ai(attack, self.model, self.vectorizer)
            status = "🚨 DETECTED" if result else "❌ MISSED"
            self.output_text.insert(tk.END, f"{status}: {attack}\n")

        self.output_text.insert(tk.END, "\n✅ Test complete.\n")

# -------------------------------
# Start GUI
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CommandInjectionGUI(root)
    root.mainloop()

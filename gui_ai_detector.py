
import joblib
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import threading
import time

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
# Train New Model
# -------------------------------
def train_new_model_from_csv(csv_path, log_widget, progressbar, root):
    try:
        progressbar["value"] = 0
        log_widget.insert(tk.END, "📄 Loading CSV data...
")
        log_widget.see(tk.END)
        root.update()

        df = pd.read_csv(csv_path)
        time.sleep(0.5)
        progressbar["value"] = 20

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        X = df['text']
        y = df['label']

        log_widget.insert(tk.END, "🧠 Vectorizing text data...
")
        log_widget.see(tk.END)
        root.update()
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        time.sleep(0.5)
        progressbar["value"] = 50

        log_widget.insert(tk.END, "🤖 Training logistic regression model...
")
        log_widget.see(tk.END)
        root.update()
        model = LogisticRegression()
        model.fit(X_vec, y)
        time.sleep(0.5)
        progressbar["value"] = 80

        log_widget.insert(tk.END, "💾 Saving model and vectorizer...
")
        log_widget.see(tk.END)
        joblib.dump(model, "command_injection_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        progressbar["value"] = 100

        log_widget.insert(tk.END, "✅ Training complete. Model updated.
")
        log_widget.see(tk.END)

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

        self.label = tk.Label(root, text="Enter hostname or input string:", font=("Arial", 12))
        self.label.pack(pady=10)

        self.input_entry = tk.Entry(root, width=50, font=("Arial", 12))
        self.input_entry.pack(pady=5)

        self.detect_button = tk.Button(root, text="Detect Injection", command=self.run_detection, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.detect_button.pack(pady=10)

        self.output_text = tk.Text(root, height=5, width=70, font=("Arial", 10))
        self.output_text.pack(pady=10)

        self.train_button = tk.Button(root, text="Train New Model from CSV", command=self.load_csv_and_train_gui, bg="#007BFF", fg="white", font=("Arial", 11))
        self.train_button.pack(pady=10)

        self.progressbar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progressbar.pack(pady=5)

        self.log_output = tk.Text(root, height=6, width=70, font=("Courier", 9), bg="#f0f0f0")
        self.log_output.pack(pady=5)

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
        self.output_text.insert(tk.END, f"Original Input:
{result}

")

        adv_input = generate_adversarial_input(user_input)
        if adv_input != user_input:
            is_adv_attack = detect_with_ai(adv_input, self.model, self.vectorizer)
            adv_result = "⚠️ Detected in adversarial form!" if is_adv_attack else "❌ Bypassed detection in adversarial form."
            self.output_text.insert(tk.END, f"Adversarial Variant:
{adv_result}
")

    def load_csv_and_train_gui(self):
        filepath = filedialog.askopenfilename(title="Select CSV Training File", filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

        def training_thread():
            train_new_model_from_csv(filepath, self.log_output, self.progressbar, self.root)
            self.model, self.vectorizer = load_model_and_vectorizer()

        threading.Thread(target=training_thread).start()

# -------------------------------
# Start GUI
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CommandInjectionGUI(root)
    root.mainloop()

import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Load Models and Vectorizers
# -------------------------------
def load_models():
    try:
        # Command Injection Model
        cmd_model = joblib.load("command_injection_model.pkl")
        cmd_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # Image Attack Detection Model (e.g., ResNet50)
        img_model = load_model("image_detection_model.h5")
        
        return cmd_model, cmd_vectorizer, img_model
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"Model not found: {str(e)}")
        return None, None, None

# -------------------------------
# Detect Command Injection (Text)
# -------------------------------
def detect_command_injection(input_string, model, vectorizer):
    input_vector = vectorizer.transform([input_string])
    prediction = model.predict(input_vector)
    return prediction[0] == 1

# -------------------------------
# Detect Adversarial Images
# -------------------------------
def detect_adversarial_image(img_path, model, threshold=0.8):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0  # Normalize
        x = np.expand_dims(x, axis=0)

        # Get prediction and confidence
        preds = model.predict(x)
        confidence = np.max(preds)
        
        # Check for low confidence (possible adversarial)
        if confidence < 0.5:
            return True, f"⚠️ Low confidence ({confidence:.2f})"
            
        # Generate adversarial example (FGSM)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = model(x)
            loss = loss_object(tf.one_hot([np.argmax(preds)], depth=1000), prediction)
        gradient = tape.gradient(loss, x)
        adv_x = x + 0.05 * tf.sign(gradient)
        adv_x = tf.clip_by_value(adv_x, 0, 1)

        # Compare SSIM
        ssim_score = ssim(x[0], adv_x[0], multichannel=True, channel_axis=-1, data_range=1.0)
        if ssim_score < threshold:
            return True, f"⛔ Adversarial detected (SSIM: {ssim_score:.2f})"
        else:
            return False, "✅ Image appears natural."
            
    except Exception as e:
        return False, f"Error: {str(e)}"

# -------------------------------
# GUI Application
# -------------------------------
class SecurityDetectorGUI:
    def __init__(self, root):
        self.cmd_model, self.cmd_vectorizer, self.img_model = load_models()
        self.root = root
        self.root.title("AI Security Detector")
        self.root.geometry("800x600")
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Command Injection
        self.tab1 = tk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Command Injection")
        self.setup_command_injection_tab()
        
        # Tab 2: Image Attack Detection
        self.tab2 = tk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Image Attack Detection")
        self.setup_image_detection_tab()
        
    def setup_command_injection_tab(self):
        # Command Injection UI
        label = tk.Label(self.tab1, text="Enter input string:", font=("Arial", 12))
        label.pack(pady=10)
        
        self.cmd_entry = tk.Entry(self.tab1, width=50, font=("Arial", 12))
        self.cmd_entry.pack(pady=5, padx=10, fill="x")
        
        self.cmd_detect_btn = tk.Button(self.tab1, text="Detect", command=self.run_cmd_detection)
        self.cmd_detect_btn.pack(pady=10)
        
        self.cmd_output = tk.Text(self.tab1, height=8, width=70, font=("Courier", 11))
        self.cmd_output.pack(pady=10, padx=10, fill="both", expand=True)
    
    def setup_image_detection_tab(self):
        # Image Detection UI
        self.img_label = tk.Label(self.tab2, text="Upload an image to check for adversarial attacks:", font=("Arial", 12))
        self.img_label.pack(pady=10)
        
        self.img_upload_btn = tk.Button(self.tab2, text="Browse Image", command=self.upload_image)
        self.img_upload_btn.pack(pady=5)
        
        self.img_display = tk.Label(self.tab2)
        self.img_display.pack(pady=10)
        
        self.img_result = tk.Label(self.tab2, text="", font=("Arial", 14, "bold"))
        self.img_result.pack(pady=10)
    
    def run_cmd_detection(self):
        input_text = self.cmd_entry.get()
        if not input_text:
            self.cmd_output.delete(1.0, tk.END)
            self.cmd_output.insert(tk.END, "Please enter some text to analyze.")
            return

        if self.cmd_model and self.cmd_vectorizer:
            is_injection = detect_command_injection(input_text, self.cmd_model, self.cmd_vectorizer)
            self.cmd_output.delete(1.0, tk.END)
            if is_injection:
                self.cmd_output.insert(tk.END, "Result: ⛔ DANGER\n\nPotential command injection attack detected!")
                self.cmd_output.tag_add("danger", "1.8", "1.14")
                self.cmd_output.tag_config("danger", foreground="red", font=("Arial", 12, "bold"))
            else:
                self.cmd_output.insert(tk.END, "Result: ✅ SAFE\n\nNo command injection patterns were found.")
                self.cmd_output.tag_add("safe", "1.8", "1.12")
                self.cmd_output.tag_config("safe", foreground="green", font=("Arial", 12, "bold"))
    
    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            try:
                img = Image.open(filepath)
                img.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                self.img_display.config(image=img_tk)
                self.img_display.image = img_tk
                
                if self.img_model:
                    is_adversarial, msg = detect_adversarial_image(filepath, self.img_model)
                    color = "red" if is_adversarial else "green"
                    self.img_result.config(text=msg, fg=color)
                else:
                    self.img_result.config(text="Image model not loaded.", fg="orange")
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not load or process the image: {e}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityDetectorGUI(root)
    root.mainloop()
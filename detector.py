import os
import threading
import io
import joblib
import requests  # <<< NEW
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, callbacks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ---------------- CONFIG ----------------
TEXT_MODEL_TYPES = ["command", "xss", "sql"]
WIN_W, WIN_H = 650, 540
MODEL_DIR = os.getcwd()

IMG_PREPROC_SIZE = (224, 224)
IMG_THUMB_SIZE = (220, 220)

DEFAULT_IMG_MIN_CONF = 0.60
ATTACK_CLASSES = ['adversarial', 'attack', 'malicious']

IMAGE_SCAN_API_URL = "http://localhost:5001/scan-image"  # <<< NEW


# ---------------- UTILS ----------------
def get_model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)

def file_exists(filename: str) -> bool:
    return os.path.exists(get_model_path(filename))


# ---------------- TEXT MODEL UTILS ----------------
def train_security_model(csv_path, model_name_prefix):
    try:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns.")
        X = df["text"].astype(str)
        y = df["label"].astype(int)

        pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
        pipe.fit(X, y)

        joblib.dump(pipe.named_steps["logisticregression"], get_model_path(f"{model_name_prefix}_model.pkl"))
        joblib.dump(pipe.named_steps["tfidfvectorizer"], get_model_path(f"{model_name_prefix}_vectorizer.pkl"))
        return True, f"{model_name_prefix.upper()} model trained and saved."
    except Exception as e:
        return False, f"Error training {model_name_prefix}: {e}"

def load_text_models():
    models_dict = {}
    for name in TEXT_MODEL_TYPES:
        try:
            m = joblib.load(get_model_path(f"{name}_model.pkl"))
            v = joblib.load(get_model_path(f"{name}_vectorizer.pkl"))
            models_dict[name] = (m, v)
        except Exception:
            models_dict[name] = (None, None)
    return models_dict

def detect_text_attack(input_string, model, vectorizer):
    X = vectorizer.transform([input_string])
    return model.predict(X)[0] == 1


# ---------------- IMAGE MODEL UTILS ----------------
def load_image_model():
    try:
        return load_model(get_model_path("image_detection_model.h5"))
    except Exception:
        return None

def build_image_generators(data_dir,
                           image_size=(224, 224),
                           batch_size=32,
                           val_split=0.2,
                           seed=123):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split,
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        seed=seed,
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        seed=seed,
    )
    return train_gen, val_gen

def build_transfer_model(num_classes, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def retrain_image_model(data_dir, epochs=10, image_size=(224, 224), batch_size=32):
    try:
        train_gen, val_gen = build_image_generators(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        num_classes = train_gen.num_classes

        model = build_transfer_model(num_classes=num_classes, input_shape=(*image_size, 3))

        es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
        rlrop = callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[es, rlrop],
            verbose=1,
        )

        model.save(get_model_path("image_detection_model.h5"))
        return True, "Image model trained and saved.", model, train_gen.class_indices
    except Exception as e:
        return False, f"Image training failed: {e}", None, None


# ---------------- IMAGE DETECTION (LOCAL ML) ----------------
def detect_adversarial_image(
    img_path, model, class_names, min_confidence
):
    """
    Local model detection: (is_attack_bool, message_str)
    Attack if:
      - max probability < min_confidence
      - predicted class in ATTACK_CLASSES
    """
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMG_PREPROC_SIZE)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        x = np.expand_dims(arr, axis=0)

        preds = model.predict(x, verbose=0)
        conf = float(np.max(preds))
        idx = int(np.argmax(preds))
        pred_name = class_names[idx].lower() if idx < len(class_names) else ""

        if conf < min_confidence:
            return True, "⚠ Suspicious image (low confidence)."
        if pred_name in ATTACK_CLASSES:
            return True, "⛔ Adversarial image detected."
        return False, "✅ Image appears safe."
    except Exception as e:
        return True, f"Error: {e}"


# ---------------- IMAGE DETECTION (API) ----------------  <<< NEW
def scan_image_via_api(img_path):
    """
    Sends image to external heuristic API and returns (is_attack, message).
    """
    try:
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f, "application/octet-stream")}
            resp = requests.post(IMAGE_SCAN_API_URL, files=files, timeout=10)
        if resp.status_code != 200:
            return True, f"API error: HTTP {resp.status_code}"
        data = resp.json()
        is_attack = bool(data.get("is_attack", False))
        reason = data.get("reason", "No reason provided.")
        return is_attack, reason
    except Exception as e:
        return True, f"API request failed: {e}"


# ---------------- GUI ----------------
class SecurityDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Security Detector")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="#f2f4f7")

        # Models
        self.text_models = load_text_models()
        self.img_model = load_image_model()
        self.img_class_names = []  # populated after training

        # Confidence threshold
        self.img_min_conf_var = tk.DoubleVar(value=DEFAULT_IMG_MIN_CONF)

        # Image backend selector  <<< NEW
        self.image_backend_var = tk.StringVar(value="local")  # "local" or "api"

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)

        # Header
        self.header_frame = tk.Frame(root, bg="#f2f4f7")
        self.header_frame.pack(fill="x", pady=5)

        tk.Label(self.header_frame, text="Model Directory:", bg="#f2f4f7",
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=5)
        self.dir_label = tk.Label(self.header_frame, text=MODEL_DIR, bg="#f2f4f7", fg="blue")
        self.dir_label.pack(side="left", padx=5)
        ttk.Button(self.header_frame, text="Change", command=self._change_dir).pack(side="left", padx=5)

        # Status Panel
        self.status_frame = tk.Frame(root, bg="#f2f4f7")
        self.status_frame.pack(fill="x", pady=5)
        self.status_labels = {}
        for name in ["Command", "XSS", "SQL", "Image"]:
            lbl = tk.Label(self.status_frame, text=f"{name}: ❌", bg="#f2f4f7", font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left", padx=10)
            self.status_labels[name.lower()] = lbl

        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_text = tk.Frame(self.notebook, bg="#f2f4f7")
        self.notebook.add(self.tab_text, text="Text Detection")
        self._setup_text_tab()

        self.tab_img = tk.Frame(self.notebook, bg="#f2f4f7")
        self.notebook.add(self.tab_img, text="Image Detection")
        self._setup_img_tab()

        self._update_status()

    # ----- STATUS -----
    def _update_status(self):
        for t in TEXT_MODEL_TYPES:
            have_model = file_exists(f"{t}_model.pkl")
            have_vec = file_exists(f"{t}_vectorizer.pkl")
            state = "✅" if have_model and have_vec else "❌"
            self.status_labels[t].config(text=f"{t.capitalize()}: {state}")
        state_img = "✅" if file_exists("image_detection_model.h5") else "❌"
        self.status_labels["image"].config(text=f"Image: {state_img}")

    def _change_dir(self):
        global MODEL_DIR
        path = filedialog.askdirectory()
        if path:
            MODEL_DIR = path
            self.dir_label.config(text=MODEL_DIR)
            self.text_models = load_text_models()
            self.img_model = load_image_model()
            self._update_status()

    # ----- TEXT TAB -----
    def _setup_text_tab(self):
        tk.Label(self.tab_text, text="Enter input string:", font=("Segoe UI", 10, "bold"),
                 bg="#f2f4f7").pack(pady=5)
        self.text_input = tk.Entry(self.tab_text, width=45, font=("Segoe UI", 11))
        self.text_input.pack(pady=5)

        tk.Label(self.tab_text, text="Select Detection Type:", font=("Segoe UI", 10),
                 bg="#f2f4f7").pack(pady=2)
        self.detection_type = tk.StringVar(value="command")
        ttk.Combobox(
            self.tab_text,
            textvariable=self.detection_type,
            values=TEXT_MODEL_TYPES,
            state="readonly",
            font=("Segoe UI", 10),
        ).pack(pady=5)

        ttk.Button(self.tab_text, text="Detect Attack",
                   command=self.run_text_detection).pack(pady=8)

        frame_train = tk.Frame(self.tab_text, bg="#f2f4f7")
        frame_train.pack(pady=4)
        for i, t in enumerate(TEXT_MODEL_TYPES):
            ttk.Button(frame_train, text=f"Train {t.capitalize()}",
                       command=lambda x=t: self._start_text_training(x)).grid(row=0, column=i, padx=3)

        self.text_output = scrolledtext.ScrolledText(self.tab_text, height=8, width=60,
                                                     font=("Consolas", 10), bg="#1e1e1e", fg="#ffffff")
        self.text_output.pack(padx=10, pady=10, fill="both", expand=True)

    def run_text_detection(self):
        text = self.text_input.get().strip()
        self.text_output.delete("1.0", tk.END)
        if not text:
            self.text_output.insert(tk.END, "⚠ Please enter some text.\n")
            return
        t = self.detection_type.get()
        model, vec = self.text_models.get(t, (None, None))
        if not model or not vec:
            self.text_output.insert(tk.END, f"No trained {t} model.\n")
            return
        mal = detect_text_attack(text, model, vec)
        self.text_output.insert(tk.END, "⛔ ATTACK DETECTED!\n" if mal else "✅ SAFE.\n")

    def _start_text_training(self, model_type):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        messagebox.showinfo("Training", f"Training {model_type}...")
        def worker():
            success, msg = train_security_model(path, model_type)
            self.root.after(0, lambda: self._finish_text_training(success, msg))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_text_training(self, success, msg):
        messagebox.showinfo("Training Result", msg)
        if success:
            self.text_models = load_text_models()
            self._update_status()

    # ----- IMAGE TAB -----
    def _setup_img_tab(self):
        tk.Label(self.tab_img, text="Upload image:", font=("Segoe UI", 10, "bold"),
                 bg="#f2f4f7").pack(pady=5)
        ttk.Button(self.tab_img, text="Browse Image", command=self._upload_image).pack(pady=5)
        ttk.Button(self.tab_img, text="Train Image Model from Folder", command=self._start_img_training).pack(pady=5)

        # Backend selector  <<< NEW
        backend_frame = tk.Frame(self.tab_img, bg="#f2f4f7")
        backend_frame.pack(pady=5)
        tk.Label(backend_frame, text="Detection Backend:", bg="#f2f4f7").grid(row=0, column=0, padx=4)
        ttk.Radiobutton(backend_frame, text="Local Model", value="local",
                        variable=self.image_backend_var).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(backend_frame, text="API", value="api",
                        variable=self.image_backend_var).grid(row=0, column=2, padx=4)

        # Threshold slider
        slider_frame = tk.Frame(self.tab_img, bg="#f2f4f7")
        slider_frame.pack(pady=5)
        tk.Label(slider_frame, text="Attack Confidence Threshold:", bg="#f2f4f7",
                 font=("Segoe UI", 9)).grid(row=0, column=0, padx=4, sticky="w")
        self.slider_label_val = tk.Label(slider_frame, text="60%", bg="#f2f4f7")
        self.slider_label_val.grid(row=0, column=1, padx=4, sticky="w")
        self.img_threshold_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=200,
            showvalue=False,
            command=self._on_threshold_change,
            bg="#f2f4f7",
            highlightthickness=0,
        )
        self.img_threshold_slider.set(int(DEFAULT_IMG_MIN_CONF * 100))
        self.img_threshold_slider.grid(row=1, column=0, columnspan=2, pady=2)

        self.img_display = tk.Label(self.tab_img, bg="#f2f4f7")
        self.img_display.pack(pady=10)

        self.img_result = tk.Label(self.tab_img, text="", font=("Segoe UI", 12, "bold"), bg="#f2f4f7")
        self.img_result.pack(pady=5)

    def _on_threshold_change(self, val):
        pct = int(float(val))
        self.slider_label_val.config(text=f"{pct}%")
        self.img_min_conf_var.set(pct / 100.0)

    def _upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return

        # Show image
        img = Image.open(path)
        img.thumbnail(IMG_THUMB_SIZE)
        img_tk = ImageTk.PhotoImage(img)
        self.img_display.config(image=img_tk)
        self.img_display.image = img_tk

        # Decide backend
        backend = self.image_backend_var.get()
        if backend == "api":
            is_adv, msg = scan_image_via_api(path)
            color = "red" if is_adv else "green"
            self.img_result.config(text=msg, fg=color)
            return

        # Local model path
        if not self.img_model:
            self.img_result.config(text="No image model loaded", fg="orange")
            return

        class_names = self.img_class_names if self.img_class_names else ["class_0", "class_1"]
        is_adv, msg = detect_adversarial_image(
            path,
            self.img_model,
            class_names,
            min_confidence=self.img_min_conf_var.get(),
        )
        self.img_result.config(text=msg, fg=("red" if is_adv else "green"))

    def _start_img_training(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        messagebox.showinfo("Training", "Image model training...")
        def worker():
            success, msg, model, class_idx = retrain_image_model(folder)
            self.root.after(0, lambda: self._finish_img_training(success, msg, model, class_idx))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_img_training(self, success, msg, model, class_idx):
        messagebox.showinfo("Training Result", msg)
        if success and model:
            self.img_model = model
            if class_idx:
                names = [None] * len(class_idx)
                for name, idx in class_idx.items():
                    names[idx] = name
                self.img_class_names = [n if n is not None else "" for n in names]
            self._update_status()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityDetectorGUI(root)
    root.mainloop()

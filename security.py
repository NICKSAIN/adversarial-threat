import os
import threading
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, callbacks
from skimage.metrics import structural_similarity as ssim  # SSIM
import requests  # For calling the API

# ---- Flask for embedded heuristic image API ----
from flask import Flask, request, jsonify
import webbrowser

# ---------------- CONFIGURATION ----------------
TEXT_MODEL_TYPES = ["command", "xss", "sql"]

WIN_W, WIN_H = 650, 620  # slightly taller to fit API button

MODEL_DIR = os.getcwd()

IMG_PREPROC_SIZE = (224, 224)
IMG_THUMB_SIZE = (220, 220)

DEFAULT_IMG_MIN_CONF = 0.60  # fixed threshold
ATTACK_CLASSES = ['adversarial', 'attack', 'malicious']

# API endpoint (local flask started from GUI)
IMAGE_SCAN_API_URL = "http://localhost:5001/scan-image"


# ---------------- UTILITY FUNCTIONS ----------------
def get_model_path(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)

def file_exists(filename: str) -> bool:
    return os.path.exists(get_model_path(filename))


# ---------------- TEXT MODEL UTILS ----------------
def train_security_model(csv_path: str, model_name_prefix: str):
    """
    Train TF-IDF + LogisticRegression model from CSV.
    CSV must have: text,label
    """
    try:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")
        X = df["text"].astype(str)
        y = df["label"].astype(int)

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000, solver='liblinear')
        )
        pipeline.fit(X, y)

        # Save components
        joblib.dump(
            pipeline.named_steps["logisticregression"],
            get_model_path(f"{model_name_prefix}_model.pkl")
        )
        joblib.dump(
            pipeline.named_steps["tfidfvectorizer"],
            get_model_path(f"{model_name_prefix}_vectorizer.pkl")
        )
        return True, f"✅ {model_name_prefix.upper()} model trained and saved successfully."
    except Exception as e:
        return False, f"❌ Error training {model_name_prefix} model: {e}"

def load_text_models() -> dict:
    models_dict = {}
    for name in TEXT_MODEL_TYPES:
        try:
            model = joblib.load(get_model_path(f"{name}_model.pkl"))
            vectorizer = joblib.load(get_model_path(f"{name}_vectorizer.pkl"))
            models_dict[name] = (model, vectorizer)
        except Exception:
            models_dict[name] = (None, None)
    return models_dict

def detect_text_attack(input_string: str, model, vectorizer) -> bool:
    X_transformed = vectorizer.transform([input_string])
    prediction = model.predict(X_transformed)
    return prediction[0] == 1


# ---------------- IMAGE MODEL UTILS ----------------
def load_image_model():
    try:
        return load_model(get_model_path("image_detection_model.h5"))
    except Exception:
        return None

def build_image_generators(data_dir: str, image_size: tuple, batch_size: int,
                           val_split: float = 0.2, seed: int = 123):
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

def build_transfer_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def retrain_image_model(data_dir: str, epochs: int = 10,
                        image_size: tuple = (224, 224), batch_size: int = 32):
    try:
        train_gen, val_gen = build_image_generators(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )
        num_classes = train_gen.num_classes
        if num_classes == 0:
            raise ValueError("No image classes found in the specified directory.")
        if train_gen.samples == 0:
            raise ValueError("No training images found.")

        model = build_transfer_model(num_classes=num_classes, input_shape=(*image_size, 3))

        early_stopping = callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        )
        reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        model.save(get_model_path("image_detection_model.h5"))
        return True, "✅ Image model trained and saved successfully.", model, train_gen.class_indices
    except Exception as e:
        return False, f"❌ Image training failed: {e}", None, None


# ---------------- LOCAL ADVERSARIAL IMAGE CHECK ----------------
def detect_adversarial_image(img_path: str, model, class_names: list,
                             min_confidence: float):
    """
    Local adversarial check using:
      - Confidence threshold
      - Predicted attack-class name match
      - SSIM drop after FGSM-like perturbation
    Returns (is_attack_bool, message_str)
    """
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMG_PREPROC_SIZE)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        x_original = np.expand_dims(arr, axis=0)

        x_tensor = tf.convert_to_tensor(x_original, dtype=tf.float32)
        preds = model.predict(x_tensor, verbose=0)
        confidence = float(np.max(preds))
        predicted_idx = int(np.argmax(preds))
        predicted_class_name = (
            class_names[predicted_idx].lower()
            if class_names and predicted_idx < len(class_names)
            else ""
        )

        print(f"\n--- Image Detection Debug Info for {os.path.basename(img_path)} ---")
        print(f"Confidence: {confidence:.2f}")
        print(f"Predicted Class: '{predicted_class_name}' (Index: {predicted_idx})")
        print(f"Min Confidence Threshold: {min_confidence:.2f}")
        print(f"Attack Classes: {ATTACK_CLASSES}")

        if confidence < min_confidence:
            print("-> Low confidence.")
            return True, "⚠ Suspicious (low confidence)."

        if predicted_class_name in ATTACK_CLASSES:
            print(f"-> Predicted attack class '{predicted_class_name}'.")
            return True, "⛔ Adversarial (predicted attack class)."

        # SSIM after FGSM-like perturbation
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            prediction_logits = model(x_tensor, training=False)
            loss = loss_object(tf.constant([predicted_idx]), prediction_logits)

        gradient = tape.gradient(loss, x_tensor)
        if gradient is None:
            print("-> Gradient not computable.")
            return False, "✅ Image appears safe (no gradient)."

        epsilon = 0.05
        adv_x_tensor = x_tensor + epsilon * tf.sign(gradient)
        adv_x_tensor = tf.clip_by_value(adv_x_tensor, 0, 1)
        adv_x_numpy = adv_x_tensor[0].numpy()

        ssim_threshold = 0.98
        try:
            ssim_score = ssim(x_original[0], adv_x_numpy, channel_axis=2, data_range=1.0)
        except TypeError:
            ssim_score = ssim(x_original[0], adv_x_numpy, multichannel=True, data_range=1.0)

        print(f"SSIM Score: {ssim_score:.4f} (Threshold: {ssim_threshold:.2f})")
        if ssim_score < ssim_threshold:
            print("-> SSIM below threshold.")
            return True, "⛔ Adversarial (SSIM check failed)."

        print("-> Image safe.")
        return False, "✅ Image appears safe."
    except Exception as e:
        print(f"Error during image detection: {e}")
        return True, f"❌ Error during image detection: {e}"


# ---------------- EMBEDDED HEURISTIC IMAGE API ----------------
_api_app = Flask(__name__)
_api_running = False
_api_lock = threading.Lock()

@_api_app.route("/scan-image", methods=["POST"])
def _scan_image_api():
    """
    Lightweight heuristic scan (non-AI):
    - Ensure valid image
    - Reject tiny images (<50px)
    - Flag high noise (std > 90)
    """
    try:
        if "file" not in request.files:
            return jsonify({"is_attack": True, "reason": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"is_attack": True, "reason": "Empty filename"}), 400

        tmp_name = f"temp_upload_{os.getpid()}_{file.filename}"
        file.save(tmp_name)

        try:
            img = Image.open(tmp_name).convert("RGB")
            arr = np.asarray(img)
            h, w, _ = arr.shape
        finally:
            try:
                os.remove(tmp_name)
            except OSError:
                pass

        # Heuristics
        if h < 50 or w < 50:
            return jsonify({"is_attack": True, "reason": "Image too small"}), 200
        noise = float(np.std(arr))
        if noise > 90.0:
            return jsonify({"is_attack": True, "reason": f"High noise ({noise:.1f})"}), 200

        return jsonify({"is_attack": False, "reason": "Safe image"}), 200
    except Exception as e:
        return jsonify({"is_attack": True, "reason": f"Error scanning: {e}"}), 500

def start_api_server():
    """
    Start the embedded Flask server in a background thread (once).
    """
    global _api_running
    with _api_lock:
        if _api_running:
            return False
        _api_running = True

    def _run():
        # use_reloader=False prevents double-start
        _api_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

    threading.Thread(target=_run, daemon=True).start()
    return True


def scan_image_via_api(img_path: str):
    """
    Client helper: send image to local API, return (is_attack, reason).
    """
    try:
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f, "application/octet-stream")}
            resp = requests.post(IMAGE_SCAN_API_URL, files=files, timeout=10)
        if resp.status_code != 200:
            return True, f"API error: HTTP {resp.status_code}"
        data = resp.json()
        return bool(data.get("is_attack", False)), data.get("reason", "No reason")
    except Exception as e:
        return True, f"API request failed: {e}"


# ---------------- GUI APPLICATION ----------------
class SecurityDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Security Detector")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="#f2f4f7")

        self.text_models = load_text_models()
        self.img_model = load_image_model()
        self.img_class_names = []  # set after training

        self.img_min_conf = DEFAULT_IMG_MIN_CONF

        # Image detection backend: "local" or "api"
        self.image_backend_var = tk.StringVar(value="local")

        # Styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TLabel", background="#f2f4f7")
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"))

        # Header
        self.header_frame = tk.Frame(root, bg="#f2f4f7")
        self.header_frame.pack(fill="x", pady=5)

        tk.Label(self.header_frame, text="Model Directory:", bg="#f2f4f7",
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=5)
        self.dir_label = tk.Label(self.header_frame, text=MODEL_DIR, bg="#f2f4f7",
                                  fg="blue", font=("Segoe UI", 9))
        self.dir_label.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(self.header_frame, text="Change", command=self._change_model_dir).pack(side="left", padx=5)

        # Status
        self.status_frame = tk.Frame(root, bg="#f2f4f7")
        self.status_frame.pack(fill="x", pady=5)
        self.status_labels = {}
        for name in TEXT_MODEL_TYPES + ["image"]:
            lbl = tk.Label(self.status_frame, text=f"{name.capitalize()}: ❌",
                           bg="#f2f4f7", font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left", padx=8)
            self.status_labels[name] = lbl

        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Text tab
        self.tab_text = tk.Frame(self.notebook, bg="#f2f4f7")
        self.notebook.add(self.tab_text, text="Text Detection")
        self._setup_text_tab()

        # Image tab
        self.tab_img = tk.Frame(self.notebook, bg="#f2f4f7")
        self.notebook.add(self.tab_img, text="Image Detection")
        self._setup_img_tab()

        # Initial status
        self._update_status_display()

    # ----- STATUS & DIR -----
    def _update_status_display(self):
        for t in TEXT_MODEL_TYPES:
            have_model = file_exists(f"{t}_model.pkl")
            have_vec = file_exists(f"{t}_vectorizer.pkl")
            state_text = "✅ Loaded" if have_model and have_vec else "❌ Missing"
            self.status_labels[t].config(
                text=f"{t.capitalize()}: {state_text}",
                fg="green" if have_model and have_vec else "red",
            )
        have_img = file_exists("image_detection_model.h5")
        self.status_labels["image"].config(
            text=f"Image: {'✅ Loaded' if have_img else '❌ Missing'}",
            fg="green" if have_img else "red",
        )

    def _change_model_dir(self):
        global MODEL_DIR
        path = filedialog.askdirectory()
        if path:
            MODEL_DIR = path
            self.dir_label.config(text=MODEL_DIR)
            self.text_models = load_text_models()
            self.img_model = load_image_model()
            self.img_class_names = []
            self._update_status_display()
            messagebox.showinfo("Directory Changed", f"Model directory set to:\n{MODEL_DIR}")

    # ----- TEXT TAB -----
    def _setup_text_tab(self):
        tk.Label(self.tab_text, text="Enter input string to analyze:",
                 font=("Segoe UI", 10, "bold"), bg="#f2f4f7").pack(pady=5)
        self.text_input = tk.Entry(self.tab_text, width=60, font=("Segoe UI", 11), bd=2, relief="groove")
        self.text_input.pack(pady=5, padx=10)

        tk.Label(self.tab_text, text="Select Detection Type:",
                 font=("Segoe UI", 10), bg="#f2f4f7").pack(pady=2)
        self.detection_type = tk.StringVar(value="command")
        self.text_type_combobox = ttk.Combobox(
            self.tab_text,
            textvariable=self.detection_type,
            values=TEXT_MODEL_TYPES,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.text_type_combobox.pack(pady=5)

        self.detect_text_btn = ttk.Button(self.tab_text, text="Detect Attack",
                                          command=self._run_text_detection)
        self.detect_text_btn.pack(pady=8)

        frame_train = tk.Frame(self.tab_text, bg="#f2f4f7")
        frame_train.pack(pady=10)
        self.text_train_buttons = {}
        for i, t in enumerate(TEXT_MODEL_TYPES):
            btn = ttk.Button(
                frame_train, text=f"Train {t.capitalize()} Model",
                command=lambda x=t: self._start_text_training(x)
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.text_train_buttons[t] = btn

        self.text_output = scrolledtext.ScrolledText(
            self.tab_text, height=10, width=70,
            font=("Consolas", 10), bg="#1e1e1e", fg="#00ff00", bd=2, relief="sunken"
        )
        self.text_output.pack(padx=10, pady=10, fill="both", expand=True)

    def _run_text_detection(self):
        input_text = self.text_input.get().strip()
        self.text_output.delete("1.0", tk.END)
        if not input_text:
            self.text_output.insert("end", "⚠ Please enter some text.\n")
            return
        selected_type = self.detection_type.get()
        model, vectorizer = self.text_models.get(selected_type, (None, None))
        if not model or not vectorizer:
            self.text_output.insert("end", f"❌ No trained model for '{selected_type}'.\n")
            return

        self.detect_text_btn.config(state="disabled")
        self.text_type_combobox.config(state="disabled")
        self.text_output.insert("end", f"Analyzing for {selected_type.upper()} attack...\n")
        self.root.update_idletasks()

        try:
            is_attack = detect_text_attack(input_text, model, vectorizer)
            if is_attack:
                self.text_output.insert("end", f"⛔ ATTACK DETECTED! ({selected_type.upper()})\n")
            else:
                self.text_output.insert("end", f"✅ SAFE. No {selected_type.upper()} patterns found.\n")
        except Exception as e:
            self.text_output.insert("end", f"❌ Error: {e}\n")
        finally:
            self.detect_text_btn.config(state="normal")
            self.text_type_combobox.config(state="readonly")

    def _start_text_training(self, model_type: str):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        for btn in self.text_train_buttons.values():
            btn.config(state="disabled")
        self.detect_text_btn.config(state="disabled")
        self.text_type_combobox.config(state="disabled")

        self.text_output.delete("1.0", tk.END)
        self.text_output.insert("end", f"Training {model_type.upper()} model...\n")
        self.root.update_idletasks()

        threading.Thread(
            target=self._text_training_worker, args=(filepath, model_type), daemon=True
        ).start()

    def _text_training_worker(self, filepath: str, model_type: str):
        success, message = train_security_model(filepath, model_type)
        self.root.after(0, self._finish_text_training, success, message)

    def _finish_text_training(self, success: bool, message: str):
        messagebox.showinfo("Training Result", message)
        self.text_output.insert("end", f"{message}\n")
        if success:
            self.text_models = load_text_models()
        self._update_status_display()
        for btn in self.text_train_buttons.values():
            btn.config(state="normal")
        self.detect_text_btn.config(state="normal")
        self.text_type_combobox.config(state="readonly")

    # ----- IMAGE TAB -----
    def _setup_img_tab(self):
        tk.Label(self.tab_img, text="Upload image to check for adversarial attacks:",
                 font=("Segoe UI", 10, "bold"), bg="#f2f4f7").pack(pady=5)

        self.browse_img_btn = ttk.Button(self.tab_img, text="Browse Image", command=self._upload_image)
        self.browse_img_btn.pack(pady=5)

        self.train_img_btn = ttk.Button(self.tab_img, text="Train Image Model from Folder",
                                        command=self._start_img_training)
        self.train_img_btn.pack(pady=5)

        # Backend selector
        backend_frame = tk.Frame(self.tab_img, bg="#f2f4f7")
        backend_frame.pack(pady=5)
        tk.Label(backend_frame, text="Detection Backend:", bg="#f2f4f7").grid(row=0, column=0, padx=4)
        ttk.Radiobutton(backend_frame, text="Local Model", value="local",
                        variable=self.image_backend_var).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(backend_frame, text="API", value="api",
                        variable=self.image_backend_var).grid(row=0, column=2, padx=4)

        # Start API Server button
        self.start_api_btn = ttk.Button(self.tab_img, text="Start API Server", command=self._start_api)
        self.start_api_btn.pack(pady=5)

        self.img_display = tk.Label(self.tab_img, bg="#f2f4f7", bd=2, relief="sunken")
        self.img_display.pack(pady=10)

        self.img_result = tk.Label(self.tab_img, text="Upload an image to analyze.",
                                   font=("Segoe UI", 12, "bold"), bg="#f2f4f7")
        self.img_result.pack(pady=5)

    def _start_api(self):
        started = start_api_server()
        if started:
            messagebox.showinfo("API Server", "API server started on http://localhost:5001/")
            try:
                webbrowser.open("http://localhost:5001/scan-image")
            except Exception:
                pass
            # Auto-switch to API backend for user convenience
            self.image_backend_var.set("api")
        else:
            messagebox.showinfo("API Server", "API server is already running.")

    def _upload_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not filepath:
            return

        # Display thumbnail
        try:
            img = Image.open(filepath)
            img.thumbnail(IMG_THUMB_SIZE)
            img_tk = ImageTk.PhotoImage(img)
            self.img_display.config(image=img_tk)
            self.img_display.image = img_tk
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not display image: {e}")
            self.img_result.config(text="Error displaying image.", fg="red")
            self.img_display.config(image='')
            self.img_display.image = None
            return

        backend = self.image_backend_var.get()
        self.browse_img_btn.config(state="disabled")
        self.train_img_btn.config(state="disabled")
        self.start_api_btn.config(state="disabled")
        self.img_result.config(text="Analyzing image...", fg="blue")
        self.root.update_idletasks()

        threading.Thread(
            target=self._image_detection_worker, args=(filepath, backend), daemon=True
        ).start()

    def _image_detection_worker(self, filepath: str, backend: str):
        if backend == "api":
            is_adv, message = scan_image_via_api(filepath)
        else:  # local
            if not self.img_model:
                is_adv, message = True, "❌ No image model loaded."
            else:
                class_names = self.img_class_names or ["class_0", "class_1"]
                is_adv, message = detect_adversarial_image(
                    filepath, self.img_model, class_names, self.img_min_conf
                )
        self.root.after(0, self._finish_image_detection, is_adv, message)

    def _finish_image_detection(self, is_adv: bool, message: str):
        self.img_result.config(text=message, fg=("red" if is_adv else "green"))
        self.browse_img_btn.config(state="normal")
        self.train_img_btn.config(state="normal")
        self.start_api_btn.config(state="normal")

    def _start_img_training(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        self.browse_img_btn.config(state="disabled")
        self.train_img_btn.config(state="disabled")
        self.start_api_btn.config(state="disabled")
        self.img_result.config(text="Training image model...", fg="blue")
        self.root.update_idletasks()
        threading.Thread(
            target=self._image_training_worker, args=(folder_path,), daemon=True
        ).start()

    def _image_training_worker(self, folder_path: str):
        success, message, model, class_indices = retrain_image_model(folder_path)
        self.root.after(0, self._finish_img_training, success, message, model, class_indices)

    def _finish_img_training(self, success: bool, message: str, model, class_indices: dict):
        messagebox.showinfo("Training Result", message)
        self.img_result.config(text=message, fg="green" if success else "red")
        if success and model:
            self.img_model = model
            if class_indices:
                names = [None] * len(class_indices)
                for name, idx in class_indices.items():
                    if idx < len(names):
                        names[idx] = name
                self.img_class_names = [
                    n if n is not None else f"unknown_{i}" for i, n in enumerate(names)
                ]
        self._update_status_display()
        self.browse_img_btn.config(state="normal")
        self.train_img_btn.config(state="normal")
        self.start_api_btn.config(state="normal")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Optional: make TF nicer with GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    root = tk.Tk()
    app = SecurityDetectorGUI(root)
    root.mainloop()

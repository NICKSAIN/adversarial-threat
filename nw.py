import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageTk

# GUI Support Functions
def select_image_and_detect(model1, model2, class_names):
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    is_adv, msg = detect_adversarial_image_strict(file_path, model1, model2, class_names)
    img = Image.open(file_path).resize((224, 224))
    tk_img = ImageTk.PhotoImage(img)
    panel.config(image=tk_img)
    panel.image = tk_img
    result_label.config(text=msg)

# Adversarial Defense Logic
def generate_fgsm_adversaries(model, images, labels, epsilon=0.03):
    adv_images = []
    for i in range(len(images)):
        x = tf.convert_to_tensor(images[i:i+1], dtype=tf.float32)
        y = tf.convert_to_tensor([labels[i]], dtype=tf.int64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
        grad = tape.gradient(loss, x)
        adv_x = x + epsilon * tf.sign(grad)
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        adv_images.append(adv_x[0].numpy())
    return np.array(adv_images)

def prediction_entropy(pred):
    return float(-np.sum(pred * np.log(pred + 1e-10)))

def detect_adversarial_image_strict(img_path: str, model1, model2, class_names: list, min_conf: float = 0.6):
    try:
        img = load_img(img_path, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        x = np.expand_dims(arr, axis=0).astype(np.float32)

        preds1 = model1.predict(x, verbose=0)
        preds2 = model2.predict(x, verbose=0)
        idx1 = int(np.argmax(preds1))
        idx2 = int(np.argmax(preds2))
        conf1 = float(np.max(preds1))
        conf2 = float(np.max(preds2))

        name1 = class_names[idx1].lower() if idx1 < len(class_names) else ""
        name2 = class_names[idx2].lower() if idx2 < len(class_names) else ""

        if name1 in ['adversarial', 'attack', 'malicious'] or name2 in ['adversarial', 'attack', 'malicious']:
            return True, "⛔ Adversarial class detected."

        if conf1 < min_conf or conf2 < min_conf:
            return True, "⚠ Low confidence detection."

        if idx1 != idx2:
            return True, "⚠ Model prediction mismatch (ensemble defense)."

        entropy = prediction_entropy(preds1[0])
        if entropy > 1.5:
            return True, f"⚠ High entropy ({entropy:.2f})"

        x_tensor = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred_logits = model1(x_tensor, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy([idx1], pred_logits)
        grad = tape.gradient(loss, x_tensor)
        adv_x = tf.clip_by_value(x_tensor + 0.05 * tf.sign(grad), 0, 1)
        adv_np = adv_x[0].numpy()
        ssim_score = ssim(x[0], adv_np, channel_axis=2, data_range=1.0)
        if ssim_score < 0.98:
            return True, f"⛔ SSIM score low: {ssim_score:.4f}"

        return False, "✅ Image appears safe."
    except Exception as e:
        return True, f"❌ Error: {e}"

def build_dual_image_models(input_shape=(224, 224, 3), num_classes=2):
    base1 = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base2 = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")

    def wrap_base(base):
        base.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs=base.input, outputs=out)

    return wrap_base(base1), wrap_base(base2)

def retrain_dual_image_models(data_dir, image_size=(224, 224), batch_size=32, epochs=10):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    X_train, y_train = next(train_gen)
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    model_temp = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
    ])
    model_temp.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    adv_X = generate_fgsm_adversaries(model=model_temp, images=X_train, labels=y_train)

    combined_X = np.concatenate([X_train, adv_X])
    combined_y = np.concatenate([y_train, y_train])

    model1, model2 = build_dual_image_models(num_classes=train_gen.num_classes)
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model1.fit(combined_X, combined_y, validation_data=val_gen, epochs=epochs)
    model2.fit(combined_X, combined_y, validation_data=val_gen, epochs=epochs)

    model1.save("image_model1.h5")
    model2.save("image_model2.h5")

    return model1, model2, train_gen.class_indices

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Zero-Stop Adversarial Image Detector")

    panel = tk.Label(root)
    panel.pack()

    result_label = tk.Label(root, text="Load an image to check.", font=("Arial", 14))
    result_label.pack(pady=10)

    load_btn = tk.Button(root, text="Load Image", command=lambda: select_image_and_detect(
        tf.keras.models.load_model("image_model1.h5"),
        tf.keras.models.load_model("image_model2.h5"),
        list(tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory("dataset", class_mode="sparse").class_indices.keys())
    ))
    load_btn.pack(pady=20)

    root.mainloop()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
import tensorflow as tf

# Load the base ResNet50 model with ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add a custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1000, activation='softmax')(x)  # Keep 1000 to match ImageNet

# Combine into final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Save it to disk
model.save("image_detection_model.h5")
print("✅ Model saved as image_detection_model.h5")

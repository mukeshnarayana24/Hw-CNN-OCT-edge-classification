import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array


# =========================
# Configuration
# =========================
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

img_height, img_width = 224, 224
batch_size = 32
float_epochs = 10
qat_epochs = 10


# =========================
# Data Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    color_mode='grayscale' 
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    color_mode='grayscale' 
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    color_mode='grayscale' 
)


# =========================
# Model 
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================
# Train Float Model
# =========================
float_history = model.fit(
    train_generator,
    epochs=float_epochs,
    validation_data=val_generator
)

float_test_loss, float_test_acc = model.evaluate(test_generator)
print(f"\nFloat Test Accuracy: {float_test_acc:.4f}")


# =========================
# Quantization-Aware Training
# =========================
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(model)

qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

qat_model.summary()


# =========================
# Train QAT Model
# =========================
qat_history = qat_model.fit(
    train_generator,
    epochs=qat_epochs,
    validation_data=val_generator
)

qat_test_loss, qat_test_acc = qat_model.evaluate(test_generator)
print(f"\nQAT Test Accuracy: {qat_test_acc:.4f}")


# =========================
# Accuracy & Loss Plots
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(float_history.history['accuracy'], label='Float Train Acc')
plt.plot(float_history.history['val_accuracy'], label='Float Val Acc')
plt.plot(qat_history.history['accuracy'], label='QAT Train Acc')
plt.plot(qat_history.history['val_accuracy'], label='QAT Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(float_history.history['loss'], label='Float Train Loss')
plt.plot(float_history.history['val_loss'], label='Float Val Loss')
plt.plot(qat_history.history['loss'], label='QAT Train Loss')
plt.plot(qat_history.history['val_loss'], label='QAT Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# =========================
# Convert to INT8 TFLite
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model_int8_qat.tflite", "wb") as f:
    f.write(tflite_model)

print("\nINT8 TFLite model saved as model_int8_qat.tflite")


# =========================
# Sample Predictions (QAT Model)
# =========================
predictions = qat_model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32")

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nSample Predictions:")
for i in random.sample(range(len(predicted_classes)), 5):
    filename = test_generator.filepaths[i]
    pred = class_labels[predicted_classes[i][0]]
    true = class_labels[true_classes[i]]
    print(f"{os.path.basename(filename)} -> Predicted: {pred}, Actual: {true}")

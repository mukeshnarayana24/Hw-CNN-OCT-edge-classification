import tensorflow as tf
import os

# ---- Settings ----
IMG_HEIGHT = 128
IMG_WIDTH  = 512
BATCH_SIZE = 32
EPOCHS     = 8

train_dir = 'dataset/train'
test_dir  = 'dataset/test'

# ---- Datasets (auto-label from folder names: NORMAL, DME) ----
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",            # 0/1 integers
    color_mode="grayscale",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Cache + prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ---- Model (small, simple CNN) ----
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # NUM_CLASSES=2
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ---- Train ----
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

# ---- Evaluate ----
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# ---- Save as SavedModel (recommended, simple) ----
model.save("oct_saved_model")
print("SavedModel written to ./oct_saved_model")


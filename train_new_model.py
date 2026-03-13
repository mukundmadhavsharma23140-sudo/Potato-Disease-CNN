import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout,
    BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import pickle

# ===============================
# CONFIGURATION
# ===============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# ===============================
# DATA GENERATORS
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7,1.3],
    shear_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "PlantVillage_processed/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    "PlantVillage_processed/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===============================
# IMPROVED CNN ARCHITECTURE
# ===============================

model = Sequential([

    # Block 1
    Conv2D(32, (3,3), padding="same", input_shape=(224,224,3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2,2),

    # Block 2
    Conv2D(64, (3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2,2),

    # Block 3
    Conv2D(128, (3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2,2),

    # Block 4 (added for depth)
    Conv2D(256, (3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2,2),

    
     #BLock 5
    Conv2D(256, (3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2,2),

    

    # Global Average Pooling (modern replacement for Flatten)
    GlobalAveragePooling2D(),

    # Fully Connected
    Dense(128, activation="relu",
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),

    Dense(3, activation="softmax")
])

# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.0005,
    weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

# ===============================
# TRAINING
# ===============================

history_obj = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_scheduler]
)

# ===============================
# SAVE HISTORY
# ===============================

history = history_obj.history

with open("history_newmodel_plantVillage.pkl", "wb") as f:
    pickle.dump(history, f)

# ===============================
# SAVE MODEL
# ===============================

model.save("potato_disease_cnn_newmodel_plantVillage.h5")

print("Training completed and model saved successfully.")
import tensorflow as tf
import numpy as np
import random
import pickle

from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2


# ===============================
# SET RANDOM SEED
# ===============================

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# ===============================
# CONFIGURATION
# ===============================

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 40


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
# LOAD MOBILENETV2 BACKBONE
# ===============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False


# ===============================
# CUSTOM CLASSIFIER HEAD
# ===============================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

outputs = Dense(3, activation="softmax")(x)


model = Model(inputs=base_model.input, outputs=outputs)


# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=1e-4
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()


# ===============================
# CALLBACKS
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)


# ===============================
# TRAIN MODEL
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

with open("history_mobilenet_model.pkl","wb") as f:
    pickle.dump(history,f)


# ===============================
# SAVE MODEL
# ===============================

model.save("potato_mobilenet_model.h5")

print("Training complete. Model saved.")
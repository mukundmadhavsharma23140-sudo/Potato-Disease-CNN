import tensorflow as tf
import numpy as np
import random
import pickle

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation,
    Multiply, Reshape, Input, Add
)

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers


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
# CBAM ATTENTION BLOCK
# ===============================

def cbam_block(x, filters, reduction=8):

    # ----- Channel Attention -----
    avg_pool = GlobalAveragePooling2D()(x)

    dense1 = Dense(filters // reduction, activation='relu')(avg_pool)
    dense2 = Dense(filters, activation='sigmoid')(dense1)

    channel = Reshape((1,1,filters))(dense2)

    x = Multiply()([x, channel])


    # ----- Spatial Attention -----
    avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)

    concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)

    spatial = Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    x = Multiply()([x, spatial])

    return x


# ===============================
# CNN + CBAM ARCHITECTURE
# ===============================

inputs = Input(shape=(224,224,3))

# Block 1
x = Conv2D(32,(3,3),padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(2,2)(x)

# Block 2
x = Conv2D(64,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(2,2)(x)

# Block 3
x = Conv2D(128,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(2,2)(x)

# Block 4
x = Conv2D(256,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

# CBAM Attention Layer
x = cbam_block(x,256)

x = MaxPooling2D(2,2)(x)

# Block 5
x = Conv2D(256,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(2,2)(x)

# Global Pooling
x = GlobalAveragePooling2D()(x)

# Dense Layer
x = Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.001)
)(x)

x = Dropout(0.5)(x)

outputs = Dense(3,activation="softmax")(x)

model = Model(inputs,outputs)


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
# SAVE TRAINING HISTORY
# ===============================

history = history_obj.history

with open("history_cbam_model.pkl","wb") as f:
    pickle.dump(history,f)


# ===============================
# SAVE MODEL
# ===============================

model.save("potato_cbam_cnn_model.h5")

print("Training complete. Model saved.")
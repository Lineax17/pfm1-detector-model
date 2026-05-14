import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# === Configuration ===
REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = REPO_ROOT / "data" / "train_data"
VAL_DIR = REPO_ROOT / "data" / "val_data"
TEST_DIR = REPO_ROOT / "data" / "test_data"
SAVE_MODEL_PATH = REPO_ROOT / "models" / "eurosat_multilabel_mobilenetv3_unfreezed.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

SEED = 467

# === Global Tensorflow Seed ===
tf.random.set_seed(SEED)

# === Loading Data ===
def get_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

train_ds = get_dataset(TRAIN_DIR)
val_ds   = get_dataset(VAL_DIR)
test_ds  = get_dataset(TEST_DIR)

# === Save Classnames ===
train_ds_raw = get_dataset(TRAIN_DIR)
class_names = train_ds_raw.class_names

# === Prefetching & Caching ===
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# === Augmentation Layer ===
data_augmentation = tf.keras.Sequential([
    RandomRotation(0.1),             # Rotate +/- 10%
    RandomZoom(0.1),                 # Zoom +/- 10%
    RandomContrast(0.1)              # Contrast adjustment (replaces brightness)
], name="augmentation_layer")

# === Model construction: Transfer Learning with MobileNetV3Small ===
base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Last 30 Layers trainable
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Augmentation is applied during training, ignored during inference
x = preprocess_input(x) # MobileNetV3 Preprocessing
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(str(SAVE_MODEL_PATH), save_best_only=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# === Modell speichern ===
model.save(str(SAVE_MODEL_PATH))

# === Testbewertung ===
model = load_model(str(SAVE_MODEL_PATH))
loss, acc = model.evaluate(test_ds)
print(f"\n🧪 Final Test Accuracy: {acc * 100:.2f}%")
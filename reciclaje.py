import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from kagglehub import dataset_download
from PIL import Image

#todas las dependencias de kaggle
dataset_path = dataset_download("saumyamohandas/garbage-classification-image-dataset")
print("Ruta del dataset:", dataset_path)

for folder in os.listdir(dataset_path):
  print(folder)

IMG_SIZE = (224,224)
BATCH_SIZE = 32
SEED =42

#Definir DATA_DIR el path de kagglehub dowload
DATA_DIR = os.path.join(dataset_path, 'dataset')

train_data_path =os.path.join(DATA_DIR, 'Training')
test_data_path=os.path.join(DATA_DIR, 'Testing')

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_path,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_name = train_ds.class_names
print("Clases detectadas: ", class_name)

plt.figure(figsize=(10,8))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

AUTOTUNE = tf.data.AUTOTUNE

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)

base_model= tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable=False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x= base_model(inputs, training=False) #neurana base
x= tf.keras.layers.GlobalAveragePooling2D()(x) #reducir los parametros de analisis de las imagenes
x= tf.keras.layers.Dropout(0.2)(x) # va a tomar el 20% de las neuronas  y las demas no se toca
outputs = tf.keras.layers.Dense(len(class_name), activation='softmax')(x) #sacar una probablidad que esa imagen es lo que dice el modelo
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.summary()

EPOCHS = 1

callbacks = [
    tf.keras.callbacks.EarlyStopping( # detiene el entrenamiento cuando ve que ya no aprende
        monitor='val_loss',
        patience=3, #con dos epochs consecutivas no aprende el modelo no va a prender, entonces se para
        restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")

OUTPUT_DIR = Path("modelo_reciclaje_mobilenet")
OUTPUT_DIR.mkdir(exist_ok=True)

model.save(OUTPUT_DIR / "waste_mobilenet.h5")
#(HDF5 - Hierarchical Data Format version 5)
model.save(OUTPUT_DIR / "waste_mobilenet.keras")

with open(OUTPUT_DIR / "class_name.json", "w", encoding= "utf-8") as f:
    json.dump(class_name, f, ensure_ascii= False, indent=2)

print("Modelo guardado en: ", OUTPUT_DIR / "waste_mobilenet.h5")
print("Modelo guardado en: ", OUTPUT_DIR / "waste_mobilenet.keras")
print("Clases guardadas en: ", OUTPUT_DIR / "class_name.json")

test_model = tf.keras.models.load_model(
   "modelo_reciclaje_mobilenet/waste_mobilenet.h5",
   compile= False
)

print("Modelo cargado correctamente")
test_model.summary()

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Descarte"
}

def predict_image(ima_path_param):
    img = Image.open(ima_path_param).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype= np.float32) / 255.0
    #255 porque tensor flow maneja mejor las imagenes de 0 a 1
    arr = np.expand_dims(arr, axis=0)
    preds = test_model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    for idx in top3_idx:
        raw = class_name[idx]
        print(f"{LABELS_ES.get(raw, raw)}: {preds[idx] * 100:.2f}%")
    return class_name[np.argmax(preds)]

#Ejemplo
example_img_path = "vidrio.jpg"
predict_image(example_img_path)
plt.imshow(Image.open(example_img_path)); plt.title("Imagen analizada"); plt.axis("off")

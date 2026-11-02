# Clasificación de reciclaje con MobileNetV2

Autora: Jeimy Jazmín Palma Santos

## Qué hace
Entrena una red neuronal que recibe la foto de un desecho y la clasifica
en 6 tipos: cartón, vidrio, metal, papel, plástico o descarte.
Guarda el modelo entrenado para usarlo después en una app de Streamlit.

## Tecnologías
Python, TensorFlow/Keras (MobileNetV2), NumPy, Pillow, Matplotlib y kagglehub.

## Cómo funciona
1. Descarga de Kaggle un dataset de fotos de desechos separadas por carpeta.
2. Carga las fotos de entrenamiento y validación en tamaño 224x224.
3. Usa MobileNetV2 preentrenada (transfer learning) y le agrega una capa
   que decide entre las 6 clases.
4. Entrena el modelo y mide su precisión (accuracy).
5. Guarda el modelo y la lista de clases.
6. Predice la clase de una foto nueva y muestra las 3 más probables.

## Cómo se ejecuta
Se diseñó para Google Colab. Requiere:
pip install tensorflow numpy pillow matplotlib kagglehub

## Análisis de calidad
Este repositorio se analiza con SonarQube Cloud como parte del curso IS-802.

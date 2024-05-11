# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:38:00 2024

@author: User_Asus
"""

import os
import numpy as np
from PIL import Image

# Función para dividir una imagen en partes más pequeñas
def split_image(image_path, split_size):
    img = Image.open(image_path)
    width, height = img.size
    parts = []
    labels = []  # Lista para almacenar las etiquetas de clase asociadas a cada parte de la imagen
    for i in range(0, width, split_size):
        for j in range(0, height, split_size):
            box = (i, j, i+split_size, j+split_size)
            region = img.crop(box)
            parts.append(np.array(region))
            labels.append(image_path.split(os.path.sep)[-2])  # Obtener la etiqueta de clase del nombre de la carpeta
    return parts, labels

# Función para clasificar manualmente una parte de la imagen
def manual_classification(part):
    # Mostrar la parte de la imagen y esperar la entrada del usuario
    Image.fromarray(part).show()
    classification = input("Clasificación (q para Fire, w para No Fire, e para Smoke, r para Noise): ")
    return classification

# Definir la carpeta de entrada y salida
input_folder = r"C:\Users\User_Asus\Downloads\Incendio_Forestal\training"
output_folder = r"C:\Users\User_Asus\Downloads\Incendio_Forestal\nuevos_result"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar todas las imágenes en la carpeta de entrada
for class_folder in os.listdir(input_folder):
    class_folder_path = os.path.join(input_folder, class_folder)
    if os.path.isdir(class_folder_path):
        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            parts, labels = split_image(image_path, 50)
            for idx, (part, label) in enumerate(zip(parts, labels)):
                classification = manual_classification(part)
                class_output_folder = os.path.join(output_folder, classification)
                os.makedirs(class_output_folder, exist_ok=True)
                part_image = Image.fromarray(part)
                part_image.save(os.path.join(class_output_folder, f"{label}_{filename}_part_{idx}.jpg"))

# Entrenamiento de neuronas, asegurar de descargar el archivo 'tu_archivo_normalizado.csv'

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Cargar los datos desde el archivo CSV
data = pd.read_csv("tu_archivo_normalizado.csv")

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Codificar las etiquetas en un formato adecuado para la red neuronal
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Definir los rangos de parámetros
learning_rates = [0.01, 0.1, 0.5, 1.0]
momentum_values = [0.1, 0.3, 0.5, 0.7, 0.9]
neuronas_entrada = 3
descriptores = 21
neurons_range = list(range(3, 24))
epochs_range = [100, 200, 300, 400, 500]

# Inicializar variables para el mejor accuracy y los parámetros correspondientes
best_accuracy = 0.0
best_parameters = {}

# Crear listas para almacenar los resultados
results = []
accuracy_by_epoch = []

# Definir la partición con k=13
stratified_kfold = StratifiedKFold(n_splits=13, shuffle=True, random_state=42)

# Iterar sobre las particiones y entrenar modelos
for train_index, test_index in stratified_kfold.split(X, y_encoded):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    for learning_rate in learning_rates:
        for momentum in momentum_values:
            for neurons in neurons_range:
                for epochs in epochs_range:
                    model = Sequential([
                        Input(shape=(X_train.shape[1],)),  # Capa de entrada
                        Dense(neurons, activation='relu'),  # Capa oculta
                        Dense(3, activation='softmax')  # Capa de salida
                    ])

                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                    history = model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes=3), epochs=epochs, batch_size=16, verbose=0)

                    _, accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, num_classes=3), verbose=0)
                    results.append([learning_rate, momentum, 1, neurons, epochs, accuracy])
                    accuracy_by_epoch.append((neurons, epochs, history.history['accuracy'], history.history['loss']))

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters['learning_rate'] = learning_rate
                        best_parameters['momentum'] = momentum
                        best_parameters['neurons'] = neurons
                        best_parameters['epochs'] = epochs

                    print(f"Learning rate: {learning_rate}, Momentum: {momentum}, Capa: 1, Neurona: {neurons}, Época: {epochs}, Accuracy: {accuracy}")

# Convertir la lista de resultados a un DataFrame de pandas
results_df = pd.DataFrame(results, columns=['learning_rate', 'momentum', 'layer', 'neurons', 'epochs', 'accuracy'])

# Guardar los resultados en un archivo CSV
results_df.to_csv('results.csv', index=False)

# Imprimir los mejores parámetros y el mejor accuracy
print("Mejor Accuracy:", best_accuracy)
print("Mejores Parámetros:", best_parameters)

# Gráfica de número de neuronas por capa oculta
neurons_counts = [neuron[3] for neuron in results]
plt.hist(neurons_counts, bins=len(set(neurons_counts)), alpha=0.5)
plt.xlabel('Número de Neuronas')
plt.ylabel('Frecuencia')
plt.title('Número de Neuronas por Capa Oculta')
plt.show()

# Gráfica de precisión y pérdida por época
for item in accuracy_by_epoch:
    neurons, epochs, accuracy, loss = item
    plt.plot(range(1, epochs + 1), accuracy, label=f'{neurons} Neuronas, {epochs} Épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión por Época')
plt.legend()
plt.show()

plt.figure()
for item in accuracy_by_epoch:
    neurons, epochs, accuracy, loss = item
    plt.plot(range(1, epochs + 1), loss, label=f'{neurons} Neuronas, {epochs} Épocas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida por Época')
plt.legend()
plt.show()





    



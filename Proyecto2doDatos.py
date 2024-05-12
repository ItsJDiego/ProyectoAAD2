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

# ------------------------ Calculo de caracteristicas y pasarlo a csv-------------------------------------
import os
import cv2
import numpy as np
import csv
from skimage.feature import greycomatrix, greycoprops
from skimage import measure
from scipy.stats import skew, kurtosis

def extract_features(image):
    features = []

    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Media de intensidad en la imagen en escala de grises
    mean_intensity = np.mean(gray_image)
    features.append(mean_intensity)

    # 2. Desviación estándar de la intensidad en la imagen en escala de grises.
    std_intensity = np.std(gray_image)
    features.append(std_intensity)

    # Calcular el promedio, la desviación estándar y la varianza de cada componente de color (R, G, B)
    b, g, r = cv2.split(image)
    features.extend([np.mean(channel) for channel in (r, g, b)])
    features.extend([np.std(channel) for channel in (r, g, b)])
    features.extend([np.var(channel) for channel in (r, g, b)])

    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Características de color en HSV
    features.append(np.mean(hsv_image[:, :, 0]))  # Promedio de tono
    features.append(np.mean(hsv_image[:, :, 1]))  # Promedio de saturación
    features.append(np.mean(hsv_image[:, :, 2]))  # Promedio de valor

    # 4. Características de textura utilizando la matriz de co-ocurrencia de niveles de gris (GLCM)
    glcm = greycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    features.append(greycoprops(glcm, 'homogeneity').mean())  # Homogeneidad
    features.append(greycoprops(glcm, 'contrast').mean())     # Contraste

    # 5. Características de forma utilizando regionprops
    props = measure.regionprops_table(measure.label(gray_image > np.mean(gray_image)), properties=('area', 'perimeter', 'solidity'))
    features.extend([np.mean(props['area']), np.mean(props['perimeter']), np.mean(props['solidity'])])

    # 6. Características estadísticas: asimetría y curtosis
    features.append(skew(gray_image, axis=None))     # Asimetría
    features.append(kurtosis(gray_image, axis=None)) # Curtosis

    return features

# Carpeta de entrada
input_folder = r"/content/drive/MyDrive/nuevos_result"

# Definir las clases
classes = ['q', 'w', 'e']

# Crear un archivo CSV para escribir las características
csv_filename = "image_features5.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Escribir el encabezado del CSV
    header = ["Class", "Mean Intensity", "Std Intensity", "Mean R", "Mean G", "Mean B", "Std R", "Std G", "Std B", "Var R", "Var G", "Var B",
              "Promedio_tono", "Promedio_saturacion", "Promedio_valor", "Homogeneidad", "Contraste", "Area", "Perimeter", "Solidity",
              "Asimetría", "Curtosis"]
    csv_writer.writerow(header)

    # Iterar sobre las clases
    for cls in classes:
        # Obtener la carpeta de la clase actual
        class_folder = os.path.join(input_folder, cls)

        # Obtener la lista de archivos en la carpeta de la clase
        file_list = os.listdir(class_folder)

        # Iterar sobre los archivos de la clase actual
        for filename in file_list:
            # Construir la ruta completa de la imagen
            image_path = os.path.join(class_folder, filename)

            # Cargar la imagen
            image = cv2.imread(image_path)

            # Extraer características de la imagen
            image_features = extract_features(image)

            # Escribir las características en el archivo CSV junto con la clase
            row = [cls] + image_features
            csv_writer.writerow(row)

print("CSV generado exitosamente:", csv_filename)

# ------------------------ Realizar la normalización -------------------------------------

import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('image_features5.csv')

# Guardar la columna 'class' antes de normalizar
clases = df['Class']

# Eliminar la columna 'class' temporalmente antes de normalizar
df = df.drop(columns=['Class'])

# Normalizar utilizando min-max scaling
df_normalized = (df - df.min()) / (df.max() - df.min())

# Agregar la columna 'class' nuevamente
df_normalized['Class'] = clases

# Guardar el archivo CSV normalizado
df_normalized.to_csv('tu_archivo_normalizado.csv', index=False)


# --------------------Entrenamiento de neuronas, asegurar de descargar el archivo 'tu_archivo_normalizado.csv'--------------------------

import pandas as pd
import tensorflow as tf
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
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

                    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    results.append([learning_rate, momentum, 1, neurons, epochs, accuracy])
                    accuracy_by_epoch.append((neurons, epochs, history.history['accuracy'], history.history['loss']))

                    print(f"Learning rate: {learning_rate}, Momentum: {momentum}, Capa: 1, Neurona: {neurons}, Época: {epochs}, Accuracy: {accuracy}")

                    # Convertir la lista de resultados a un DataFrame de pandas y guardarlos en un archivo CSV
                    results_df = pd.DataFrame(results, columns=['learning_rate', 'momentum', 'layer', 'neurons', 'epochs', 'accuracy'])
                    results_df.to_csv('results.csv', index=False)

# Imprimir los resultados finales
print("Resultados guardados en 'results.csv'")


# ------------------------------------------------ Gráficas del punto a, b y c del inciso 6-----------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los resultados desde el archivo CSV
results_df = pd.read_csv('results.csv')

# Seleccionar las filas del DataFrame del 23 al 44
selected_results = results_df.iloc[22:44]

# Gráfico de precisión global en función del Learning Rate y Momentum
plt.figure(figsize=(10, 6))

# Crear un mapa de calor para visualizar la precisión global
accuracy_heatmap = selected_results.pivot_table(values='accuracy', index='learning_rate', columns='momentum')
sns.heatmap(accuracy_heatmap, annot=True, cmap='viridis', fmt=".3f")

plt.title('Precisión Global vs Learning Rate y Momentum')
plt.xlabel('Momentum')
plt.ylabel('Learning Rate')
plt.show()


# Gráfica de número de neuronas por capa oculta
neurons_counts = [neuron[3] for neuron in results]
plt.hist(neurons_counts, bins=len(set(neurons_counts)), alpha=0.5)
plt.xlabel('Número de Neuronas')
plt.ylabel('Frecuencia')
plt.title('Número de Neuronas por Capa Oculta')
plt.show()


# Crear una paleta de colores única para cada neurona
color_palette = plt.cm.get_cmap('viridis', len(accuracy_by_epoch))

# Gráfica de precisión por época para cada neurona
plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
for i, item in enumerate(accuracy_by_epoch):
    neurons, epochs, accuracy, loss = item
    color = color_palette(i)  # Seleccionar un color de la paleta
    plt.plot(range(1, epochs + 1), accuracy, label=f'{neurons} Neuronas, {epochs} Épocas', color=color)
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión por Época')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfica de pérdida por época para cada neurona
plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
for i, item in enumerate(accuracy_by_epoch):
    neurons, epochs, accuracy, loss = item
    color = color_palette(i)  # Seleccionar un color de la paleta
    plt.plot(range(1, epochs + 1), loss, label=f'{neurons} Neuronas, {epochs} Épocas', color=color)
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida por Época')
plt.legend()
plt.tight_layout()
plt.show()







    



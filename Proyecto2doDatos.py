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

# Características 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(image):
    features = []
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Media de intensidad en la imagen en escala de grises 
    #La media de intensidad en la imagen en escala de grises proporciona información sobre el brillo 
    # promedio de la imagen
    mean_intensity = np.mean(gray_image)
    features.append(mean_intensity)
    
    # 2. Desviación estándar de la intensidad en la imagen en escala de grises.
    # indica cuánto varía la intensidad de los píxeles en relación con la media.
    # Si es alta, indica que hay una gran variabilidad en la intensidad de los píxeles, lo que podría significar que 
    # la imagen tiene regiones muy brillantes junto con regiones muy oscuras. 
    # Si es baja, indica que la intensidad de los píxeles tiende a ser más uniforme en toda la imagen.
    std_intensity = np.std(gray_image)
    features.append(std_intensity)
    
    # Calcular el promedio, la desviación estándar y la varianza de cada componente de color (R, G, B)
    b, g, r = cv2.split(image)
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)
    
    # Agregar los valores al conjunto de características
    features.extend([mean_r, mean_g, mean_b, std_r, std_g, std_b, var_r, var_g, var_b])

    return features

# Carpeta de entrada
input_folder = r"C:\Users\User_Asus\Downloads\Incendio_Forestal\nuevos_result"

# Definir las clases
classes = ['q', 'w', 'e', 'r']

# Inicializar listas para almacenar las características de cada clase
class_features = {cls: [] for cls in classes}

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
        
        # Agregar las características a la lista de características de la clase actual
        class_features[cls].append(image_features)

# Calcular el promedio de las características para cada clase
class_means = {cls: np.mean(features, axis=0) for cls, features in class_features.items()}

# Seleccionar un número máximo de características a mostrar
max_features = 11

# Crear gráficos de barras para las características seleccionadas de cada clase
for cls, mean_features in class_means.items():
    selected_features = mean_features[:max_features]
    feature_names = ['Mean Intensity', 'Std Intensity', 'Mean R', 'Mean G', 'Mean B', 'Std R', 'Std G', 'Std B', 'Var R', 'Var G', 'Var B']
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_features)), selected_features, color='b')
    plt.title(f'Average Feature Values for Class "{cls}"')
    plt.xlabel('Feature Index')
    plt.ylabel('Average Value')
    plt.xticks(range(len(selected_features)), [f'{feature_names[i]}: {selected_features[i]:.2f}' for i in range(len(selected_features))], rotation=45)
    plt.grid(True)
    plt.show()





    



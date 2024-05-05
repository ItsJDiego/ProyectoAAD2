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

    



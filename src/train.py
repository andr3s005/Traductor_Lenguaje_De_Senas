import pickle
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Cargando datos...")

# 1. Cargar datos del CSV
data = []
labels = []

# Leemos el archivo que generaron con collection.py
try:
    with open('data/keypoints.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # La primera columna es la letra (label)
            labels.append(row[0])
            # El resto son las coordenadas (data)
            data.append([float(i) for i in row[1:]])
except FileNotFoundError:
    print("¡ERROR! No se encontró data/keypoints.csv. Ejecuta collection.py primero.")
    exit()

# Convertir a arreglos de Numpy (más rápido para la PC)
data = np.asarray(data)
labels = np.asarray(labels)

# 2. Separar datos de entrenamiento y prueba
# Usamos el 20% de los datos solo para examinarse a sí mismo y ver si aprendió bien
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# 3. Entrenar el modelo
print("Entrenando el modelo (Random Forest)...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. Probar qué tan bien aprendió
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"¡Entrenamiento completado! Precisión del modelo: {score * 100:.2f}%")

# 5. Guardar el modelo entrenado
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    print("Modelo guardado exitosamente en 'model.p'")
#Problema de clasificación. Este programa tiene más de 700 neuronas de entrada y 10 de salida (las clasificaciones)
import tensorflow as tf
#tensorflow_datasets permite realizar la descarga de cpusets de datos de entrenamiento y prueba.
import tensorflow_datasets as tfds
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Descarga del set de datos para entrenamiento y pruebas de la red (60.000 - 10.000).
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

#Disponemos de los dos conjuntos de datos por separado para poder implementarlos (10 categorias disponibles).
datos_entrenamiento, datos_prueba = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names
nombres_clases

#Normalizar datos
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas
#Normalizar los datos de entrenamiento y prueba con la función previa.
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

#Agregar a cache (guarda datos en memoria y no en disco)
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()

#Mostramos la imagen de los datos de prueba obtenidos

for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28, 28))
import matplotlib.pyplot as plt

#Dibujar
# plt.figure(figsize=(10, 10))
# for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
#     imagen = imagen.numpy().reshape((28, 28))
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(imagen, cmap=plt.cm.binary)
#     plt.xlabel(nombres_clases[etiqueta])
# plt.show()

#Se crea el modelo y se implementa una red de tipo secuencial

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #1 es blanco y negro 
    #Se agregan capas de 50 neuronas densas 
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    #Capa de salida con 10 neuronas
    tf.keras.layers.Dense(10, activation=tf.nn.softmax), #Softmax para que la salida sea una probabilidad en redes de clasificación.
])

#Compilación del modelo

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#Entrenamiento de la red en lotes para agilizar el proceso
#Variables para determinar número de entrenamiento y pruebas
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["train"].num_examples

TAMANIO_LOTE = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANIO_LOTE)
datos_prueba = datos_prueba.batch(TAMANIO_LOTE)

#Entrenamos y especificamos datos de entrenamiento de la red neuronal
#epochs indica las vueltas que daremos a los datos del dataset
import math
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch = math.ceil(num_ej_entrenamiento/TAMANIO_LOTE))

#Resultados de las perdidas per epoca
# plt.xlabel('Epoca')
# plt.ylabel('Perdida')
# plt.plot(historial.history['loss'])
# plt.show()

#Pintamos una cuadricula con varias predicciones, y marcar si fue correcta (azul) o incorrecta (roja)
import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_prueba.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)
  
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_clases[etiqueta_real]),
                                color=color)
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0, 1]) 
  etiqueta_prediccion = np.argmax(arr_predicciones)
  
  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')
  
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)

#Probar una imagen suelta
imagen = imagenes_prueba[4] #AL ser la variable imagenes_prueba solo tiene lo que se le puso en el bloque anterior 
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)
print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])

#Exportacion del modelo a h5
modelo.save('modelo_exportado.h5')
#Convertir el archivo h5 a formato de tensorflowjs
# !mkdir tfjs_target_dir
# !tensorflowjs_converter --input_format keras modelo_exportado.h5 tfjs_target_dir
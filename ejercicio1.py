import cv2
import numpy as np
import matplotlib.pyplot as plt

def histograma(imagen, M, N):
    imagen_bordes = cv2.copyMakeBorder(imagen, M//2, M//2, N//2, N//2, cv2.BORDER_REPLICATE)        #Genera una imagen con bordes. Se utiliza bordes replicados.
    img_t = np.zeros_like(imagen)                                                                   #Imagen vacia con misma forma que la original
    filas, columnas = img_t.shape                                                                   #Dimensiones de la img

    for x in range(filas):                                                          
        for y in range(columnas):                                                    
            ventana = imagen_bordes[x:x + M, y: y + N]                               #Genera ventana de la imagen con bordes
            ventana_eq = cv2.equalizeHist(ventana)                                   #Aplica eq del histograma en la ventana. Como agregamos bordes no genera problemas en los bordes originales
            img_t[x, y] = ventana_eq[M//2, N//2]                                     #Setea resultado en la imagen trasnformada en la posicion que corresponde.

    return img_t


img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img_h1 = histograma(img, 3, 3)
img_h2 = histograma(img, 20, 20)
img_h3 = histograma(img, 50, 50)

plt.figure(figsize=(12, 5))
plt.title("Imagen con detalles escondidos"), plt.axis('off')
plt.subplot(131)
plt.imshow(img_h1, cmap='gray'), plt.title("Kernel = 3x3")
plt.subplot(132)
plt.imshow(img_h2, cmap='gray'), plt.title("Kernel = 25x25")
plt.subplot(133)
plt.imshow(img_h3, cmap='gray'), plt.title("Kernel = 51x51")
plt.show()

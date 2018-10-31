import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

N=3500
#se lee la imagen del arbol
arreglo_de_imagen = imread('arbol.PNG')

# para llevar a cabo la Trasnformada de Fourier 2D se usa comando o funcion de python

arreglo_de_imagen_fourier = np.fft.fft2(arreglo_de_imagen)

# Se realiza una grafica de la transformada y la guardo sin mostrarla.
fig = plt.gcf()
plt.imshow(np.abs(arreglo_de_imagen_fourier))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('BlandonValentina_FT2D.pdf')

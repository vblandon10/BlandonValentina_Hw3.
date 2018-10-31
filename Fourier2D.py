import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.fftpack import fft, ifft

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

# Se hace un filtro que me permita eliminar el ruido periodico de la imagen

arreglo_de_imagen_fourier[np.where(arreglo_de_imagen_fourier>N)] = 0

# Para esto se crea una escala lognormal
escala_lognormal = np.log(np.abs(arreglo_de_imagen_fourier))

# grafica  de la transformada de fourier despues del proceso de filtrado en escala lognormal
#guardo la grafica sin mostrarla

plt.figure()
fig = plt.gcf()
plt.imshow(escala_lognormal)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
fig.savefig('BlandonValentina_FT2D_filtrada.pdf')

#Haga la transformada de Fourier inversa y grafique la imagen filtrada.
#guardo la grafica sin mostrarla
plt.figure()
fig = plt.gcf()
plt.imshow(np.fft.ifft2(arreglo_de_imagen_fourier).real, cmap='gray')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('BlandonValentina_Imagen_filtrada.pdf')

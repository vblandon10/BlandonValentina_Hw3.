import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

#Almaceno los datos del archivo WDBC.dat. se le pone la extension .txt ya que mi computador es mac y lo convirtio a txt
Texto = np.genfromtxt('WDBC.dat.txt', dtype='str')


# se generan las listas vacias donde se guardan por aparte los numeros y las letras
numeros = []
letras = []

# se crea un for que recorra todas las lineas del texto y que separe los datos mediante comas
for linea in Texto: # Para recorrer cada linea del texto
    variables = linea.split(',') # Split separa cada dato por las comas
    # se guardan las variables en su correspondiente posicion dependiendo si es letra o numero
    numeros.append(variables[2:])
    letras.append(variables[1])

# Se convierten las listas en arreglos y se mira que los numeros sean floats
numeros = np.array(numeros)

letras = np.array(letras)

numeros = numeros.astype(float)

# Calcular covarianza dependiendo la cantidad de varianzas y de datos.
m = np.size(numeros,1) # Cantidad de variables
n = np.size(numeros,0) # Cantidad de datos

# se Calculan promedios y  se guardan
Promedios = np.zeros(m)
for i in range(m):
    Promedios[i] = np.mean(numeros[:,i]) # se calcula el promedio de cada variable y guardarlo

def covarianza(X,Y):
    # Calcula la varianza de X y Y
    n = len(X)

    # Promedios en x y y
    mx = np.mean(X)
    my = np.mean(Y)

    return np.sum( (X-mx)*(Y-my) )/(n-1)

# Se calcula la covarianza de los datos
matrizCov = np.zeros((m,m)) # se genera una matriz de ceros.
for i in range(m):
    for j in range(m):

        # i,j son recorridos en las variables (columnas)
        Varij = covarianza(numeros[:,i], numeros[:,j])
        matrizCov[i,j] = Varij #se genera la matriz de covarianza.

print(matrizCov, '\n'*3, np.cov(numeros.T))

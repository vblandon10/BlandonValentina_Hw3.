import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

#Almaceno los datos del archivo WDBC.dat. se le pone la extension .txt ya que mi computador es mac y lo convirtio a txt
Texto = np.genfromtxt('WDBC.dat.txt', dtype='str')


# se generan las listas vacias donde se guardan por aparte los numeros y las letras
numeros = []
letras = []

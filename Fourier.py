import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

#Almacene los datos de signal.dat y de incompletos.dat

signal = np.genfromtxt('signal.dat')
incompletos = np.genfromtxt('incompletos.dat')

#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla 

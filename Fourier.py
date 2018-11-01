import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

signal = np.genfromtxt('signal.dat') # cargar datos
# los datos vienen con una columna de nans, entonces elijo solo las
# columnas que me interesan
signal = signal[:,[0,-1]]


# Graficas
plt.figure()
fig = plt.gcf()
plt.plot(signal[:,0], signal[:,1])
plt.title("Grafica datos signal")
plt.grid()
plt.xlabel('Tiempo')
plt.ylabel('y')
fig.savefig('BlandonValentina_signal.pdf')



# transformadas de Fourier, con frecuencias positivas y negativas.
signal_F = []
signal_frecuencias_positivas = []
signal_frecuencias_negativas = []
# frecuencia base para calcular el vector de frecuencias
signal_fbase = (1/2) *  1/(signal[1,0]-signal[0,0]) * (2/(len(signal)))

for i in range(len(signal)):
    n = np.arange(0, len(signal))

    # transformada de fourier con alpha*exp(theta)
    theta = -1j*2*np.pi*i*n/len(signal)
    alpha = signal[:,1]
    signal_F.append( np.sum(np.sum(alpha*np.exp(theta)) ) ) # guardar datos

    if i < len(signal)/2: # Crear vector de frecuencias
        signal_frecuencias_positivas.append( i*signal_fbase )
        signal_frecuencias_negativas.append( -i*signal_fbase
        )
# concatenar las frecuencias positivas y las negativas en el orden correspondiente
signal_frecuencias = signal_frecuencias_positivas + list(reversed(signal_frecuencias_negativas))

# Convertir a arrays de numpy
signal_frecuencias = np.array(signal_frecuencias)
signal_F = np.array(signal_F)

# Crear una copia para hacer el filtro
signal_F_copia = signal_F.copy()

# Graficas
#Haga una grafica de la transformada de Fourier y guarde dicha grafica
plt.figure()
fig = plt.gcf()
plt.semilogy(signal_frecuencias, np.abs(signal_F))
plt.title("Grafica de la transformada")
plt.grid()
plt.xlabel('Frecuencias')
plt.ylabel('Magnitud')
fig.savefig('BlandonValentina_TF.pdf')

print("No estoy usando los paquetes de fft para las frecuencias, por favor evaluar el bono")
print("Las frecuencias mas importnates de la senal se presentan en la banda de 100-400Hz con tres picos cerca de 150, 200 y 400")

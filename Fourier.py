import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

#Almacene los datos de signal.dat y de incompletos.dat

signal = np.genfromtxt('signal.dat')
incompletos = np.genfromtxt('incompletos.dat')

#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla

# columnas que me interesan
signal = signal[:,[0,-1]]
xSignal  =signal[:,0]
ySignal = signal[:,1]

#grafica
plt.figure()
fig = plt.gcf()
plt.plot(xSignal, ySignal )
plt.title("Grafica de signal")
plt.grid()
plt.xlabel('Tiempo')
plt.ylabel('y')
fig.savefig('BlandonValentina_signal.pdf')


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
        signal_frecuencias_negativas.append( -i*signal_fbase )

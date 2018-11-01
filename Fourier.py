import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

#cabe resalta que distintos paquetes como np.abs, pn.roll, np.where en stackoverflow
#np.roll es para 'rodar un arreglo'.
#np.abs es para sacar el absoluto, ya que fourier son numeros complejos.

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



#Filtro pasa bajos con frecuencia de corte
def filtrar(frecuencias, amplitudes, fc1, fc2):
    # Encontrar los armonicos fuera de las frecuencias fc1 y fc2
    altas = np.where(np.logical_or(frecuencias < fc1, fc2 < frecuencias ) )

    # Eliminar esos armonicos
    amplitudes[altas] = 0

    return amplitudes

# se filtra
signal_F_filtro = filtrar(signal_frecuencias, signal_F, -1000, 1000)

#realice la transformada inversa.

signal_F_filtro = np.roll(signal_F_filtro, int(len(signal_F_filtro)/2) )# Volver a girar los arreglos para hacer la transformada inversa
signal_filtrada = ifft(signal_F_filtro)

dkldkcmdscpsdcpok[cpowopoo[o]]

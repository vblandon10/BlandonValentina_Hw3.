import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

#cabe resalta que distintos paquetes como np.abs, pn.roll, np.where en stackoverflow
#np.roll es para 'rodar un arreglo'.
#np.abs es para sacar el absoluto, ya que fourier son numeros complejos.
#El np.where es para elegir los indices de un arreglo que cumple alguna condicion
#plt.gcf() para que le devuelva a uno la ultima figura en la que se grafico

signal = np.genfromtxt('signal.dat')
incompletos = np.genfromtxt('incompletos.dat')  # cargar datos
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

# filtrar
signal_F_filtro = filtrar(signal_frecuencias, signal_F, -1000, 1000)

# hacer transformada inversa
signal_F_filtro = np.roll(signal_F_filtro, int(len(signal_F_filtro)/2) )# Volver a girar los arreglos para hacer la transformada inversa
signal_filtrada = ifft(signal_F_filtro)


#Graficas
plt.figure()
fig = plt.gcf()
plt.plot(signal[:,0], np.real(signal_filtrada))
plt.title("grafica transformada inversa")
plt.grid()
plt.xlabel('t')
plt.ylabel('y')
fig.savefig('BlandonValentina_Filtrada.pdf')

print("La trasnformada discreta de fourier esta restrigingida a senales con una frecuencia de muestreo constante. Como los datos incompletos no cumplen este requerimiento, no se puede hacer la transformada.")


# con incompletos pasa lo mismo que con signal
incompletos = incompletos[:,[0,-1]]

# Hacer un x uniforme para las interpolaciones
n = 512
xcompleto = np.linspace(min(incompletos[:,0]), max(incompletos[:,0]), n)

# Encontrar las interpolaciones cuadrada y cubica
funcion_interpolacion = interp1d(incompletos[:,0], incompletos[:,1], kind='quadratic')
signal_cuadrado = funcion_interpolacion(xcompleto)

funcion_interpolacion = interp1d(incompletos[:,0], incompletos[:,1], kind='cubic')
signal_cubico = funcion_interpolacion(xcompleto)
# transformadas de Fourier
signal_cuadrado_F = []
signal_cubico_F = []
signal_i_frecuencias_positivas = []
signal_i_frecuencias_negativas = []
signal_i_fbase = (1/2) *  1/(xcompleto[1]-xcompleto[0]) * (2/n)

for i in range(n):
    m = np.arange(0, n)

    # transformada de fourier con alpha*exp(theta)
    theta = -1j*2*np.pi*i*m/n
    alpha_cuadrado = signal_cuadrado
    alpha_cubico = signal_cubico

    signal_cuadrado_F.append( np.sum(np.sum(alpha_cuadrado*np.exp(theta)) ) ) # guardar datos
    signal_cubico_F.append( np.sum(np.sum(alpha_cubico*np.exp(theta)) ) ) # guardar datos

    if i < n/2: # Crear vector de frecuencias
        signal_i_frecuencias_positivas.append( i*signal_i_fbase )
        signal_i_frecuencias_negativas.append( -i*signal_i_fbase )

# concatenar las frecuencias positivas y las negativas en el orden correspondiente
signal_i_frecuencias = signal_i_frecuencias_positivas + list(reversed(signal_i_frecuencias_negativas))

# Convertir a arrays de numpy
signal_i_frecuencias = np.array(signal_i_frecuencias)
signal_cuadrado_F = np.array(signal_cuadrado_F)
signal_cubico_F = np.array(signal_cubico_F)



# Graficas
plt.figure(figsize=(12,8))
fig = plt.gcf()
plt.subplot(3,1,1)
plt.semilogy(signal_frecuencias, np.abs(signal_F_copia))
plt.grid()
plt.ylabel('Magnitud')
#plt.xlabel('Frecuencias')
plt.title('Espectros de frecuencias de senal base (1) e interpolaciones con splines cuadrados (2) y cubicos(3)')

plt.subplot(3,1,2)
plt.semilogy(signal_i_frecuencias, np.abs(signal_cuadrado_F))
plt.grid()
plt.ylabel('Magnitud')
#plt.xlabel('Frecuencias')

plt.subplot(3,1,3)
plt.semilogy(signal_i_frecuencias, np.abs(signal_cubico_F))
plt.grid()
plt.xlabel('Frecuencias')
plt.ylabel('Magnitud')


fig.savefig('BlandonValentina_TF_interpola.pdf')

print("En la grafica se muestra que ambas interpolaciones aumentaron la amplitud del ruido para fercuencias supeiores a 500Hz, especialmente la interpolacion cuadratica. Esto se evidencia en que al graficar la interpolacion cuadratica se generarn pequenas ondas aparentes sobre la onda base.")
print("En la interpolacion cubica tambien se generaban pequenas ondas adicionales, pero al garantizar la continuidad de mas derivadas, se obtuvo una curva mas suave y en consecuencia se introduce menos ruido a la senal base")

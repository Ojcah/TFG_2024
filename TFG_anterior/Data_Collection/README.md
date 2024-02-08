# Recolección de Datos
<p align = justify>
Para la recolección de datos se ocupa programar tanto la Jetson TX2 como el PSoC 5 LP. La Jetson TX2 consta de 4 módulos para su programación distribuidos en los archivos:  <em>Data_collection.py, RepeatedTimer.py y Transmit_Receive.py</em>. Estos módulos corresponden a un temporizador, un comunicador, un convertidor de datos y recolector de datos. En la figura 1 se muestra la interconexión de los módulos.

![Modulos_Jetson](https://user-images.githubusercontent.com/79665536/202050952-f726effe-ac99-4b18-aa46-27967329f2dc.svg)
<p align = center>
Figura 1. Interconexión de los módulos de la Jetson TX2.

<p align = justify>
El archivo <em>Data_collection.py</em> corresponde al recolector de datos, cuya función es recopilar y reinsertar los datos enviados y recibidos vía PSoC hacia y desde el PAHM. Además, se definen dos tipos de entradas: la <em>manual</em> que, por medio de una interfaz gráfica con un deslizador permite cambiar manualmente el valor PWM a transmitir y almacenar para futuras reproducciones. Por otro lado, se tiene la entrada de <em>reproducción</em> (<i>playback</i>). Este tipo de entrada reutiliza la misma entrada de un experimento previo, para de este modo reproducir y así evaluar efecto en la salida producido por factores variantes en el tiempo del PAHM ante la misma secuencia de entrada, así como determinar el promedio y la desviación estándar de las respuestas entregadas en repeticiones experimentales. Una vez finalizado los experimentos, se ponen a disposición en un archivo .csv con el fin de utilizarlos para el entrenamiento de la RNAM.

<p align = justify>
El PAHM opera con muestreo homogéneo cada 20 ms por lo que se necesita que el código se ejecute y dé respuesta en ese lapso de tiempo concreto. Por ello, se 
implementa el archivo <em>RepeatedTimer.py</em>, el cual es necesario para llamar una función cada determinado tiempo, encargada de transmitir y recibir valores en el 
tiempo requerido por el PAHM. 

<p align = justify>
Por último, el archivo <em>Transmit_Receive.py</em> se encarga de transmitir y recibir datos hacia y desde el PSoC. Este utiliza dos funciones: la función <em>reset()</em> que limpia los búferes de entrada y salida para evitar que permanezcan datos residuales de anteriores transmisiones y perjudique la lectura de los datos. Posterior a ello, la función <em>Transmit\_Receive()</em>, la cual se encarga del proceso de transmisión referido a empaquetar el valor PWM en dos <i>bytes</i> y transmitirlo al PSoC, y el proceso de recepción referido a desempaquetar los cuatro <i>bytes</i> recibidos desde el PSoC correspondientes al valor del ángulo medido por el sensor. Estos dos valores se conservan en un arreglo para la creación del archivo .csv. Dentro de este código también se realiza la conversión de datos correspondiente a la desnormalización del valor PWM encapsulada en la función <em>desnormalizePWM()</em>.

<p align = justify>
Para el uso de ellos y recolectar datos del PAHM, se debe ejecutar el archivo Data_collection.py, el cual al ejecutarlo ocupa el nombre de un archivo de entrada si se desea utilizar el modo reproducción. En caso contrario, se utiliza el modo manual. Tambien se le debe indicar el archivo de salida, el cual corresponde al archivo donde se van a almacenar los datos recopilados del PAHM.

<p align = justify>
Ejemplo del llamado: python3 Data_collection.py --input 'Archivo_a_leer.csv' --output 'Archivo_a_escribir.csv'

<p align = justify>
Ahora bien, en cuanto al sistema PSoC, se utiliza el archivo main.c el cual contiene la lógica para que el sistema funcione. Además de ello, en la aplicación <i>PSoC Creator</i> se utiliza:
<li>Un PWM digital con resolución de 16 bits y un periodo de 250 &#956s.</li>
<li>Un decodificador de cuadratura con una resolución de 16 bits y una resolución de contador de 1.</li>
<li>Un bloque UART con tasa de transmisión de 115200, 1 bytes da datos, no se activa bit de paridad y no se hace uso del control de flujo.</li>
<li>Un temporizador, donde se configura con un periodo de 20 ms.</li>
<li> Adicionalmente, se hace uso de una interrupción, un led de ejecución y dos señales de reloj: una de 8 MHz y otra de 1 MHz.</li>

<p align = justify>
Estos bloques se conectan como se muestra en el diagrama de la figura 2, que corresponde al espacio de trabajo de <i>PSoC Creator</i>.

![psoc_imagen](https://user-images.githubusercontent.com/79665536/202053218-fc2c0841-22ec-4850-a0e2-287489b6b8dd.PNG)
<p align = center>
Diagrama de conexión de bloques en el PSoC 5 LP.

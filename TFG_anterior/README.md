# Diseño e implementación de un sistema que simule el comportamiento dinámico de una planta prototipo de control automático utilizando redes neuronales artificiales

<p align = justify>
El presente proyecto busca simular mediantes redes neuronales artificiales el comportamiento dinámico de un péndulo amortiguado a hélice (PAHM). Para el desarrollo de la red neuronal artificial mimetizadora (RNAM) se requiere la conexión de la planta, quien posee un sistema embebido PSoC 5 LP, con un sistema embebido NVIDIA Jetson TX2. Para la comunicación de los sistemas se hace uso del protocolo UART. El sistema principal es la Jetson TX2 quien transmite el valor requerido para movilizar el motor codificado en PWM y recibe el ángulo del péndulo. Estos datos son almacenados para el entrenamiento de la RNAM. Los archivos pertenecientes a la comunicación entre los sistemas embebidos se encuentran en la carpeta <em>/Data_Collection</em>, donde se explica más a detalle.

<p align = justify>
Los modelos de RNAM propuestos se ponen a prueba con un escenario tanto sintético como real. El escenario sintético, denominado como <em>RNAM_Synthetic.py</em>, utiliza el modelo matemático con el fin de evaluar si la red neuronal puede aprender el comportamiento dinámico auténtico del PAHM. Mientras el escenario real, denominado como \emph{RNAM_Real.py}, consiste en el uso de datos físicos del PAHM, donde existe influencia de la incertidumbre proporcionada por el sensor de cuadratura.

<p align = justify>
El modelado matemático de la PAHM se determina empíricamente con un porcentaje de aproximación del 84.99\%. Donde la ecuación lineal que describe el comportamiento dinámico del PAHM es:

<p align = center>
  $H(s) = \frac{\theta(s)}{V(s)} = \frac{2965}{s^2 +0.4151s+ 17.97}$  

<p align = justify>
El proceso de diseño de la RNAM tiene tres etapas: el preprocesamiento de los datos, selección e implementación del modelo neuronal y su entrenamiento, y, por último, la selección de modelo que corresponde al proceso de variar los hiperparámetros y arquitectura de la red con el fin de mejorar la respuesta o predicción. Estas etapas se resumen en la figura 1, donde el interruptor indica que no siempre se realiza variación en los hiperparámetros o arquitectura.

![Diagrama_solucion_RNAM](https://user-images.githubusercontent.com/79665536/202080098-d4d3eb9e-5302-413f-a2d5-493c76895aed.svg)
<p align = center>
Figura 1. Diagrama de etapas para la elaboración de la RNAM.

<p align = justify>
Los pasos que se siguen en el preprocesamiento de los datos se resumen en la figura 2.

![etapas_preprocesamiento](https://user-images.githubusercontent.com/79665536/202080952-93cd17b4-6dac-4b94-8391-ff2d77884d27.svg)
<p align = center>
Figura 2. Etapas del preprocesamiento del conjunto de datos.

<p align = justify>
La arquitectura de la RNAM consiste en una capa GRU y una capa densa. La capa GRU es la encargada de aprender la dinámica del sistema, donde se define el tamaño de entrada de esta capa como flexible con (None, 2). Puesto que de otro modo a todos los vectores de entrenamiento, validación y prueba se les exigiría tener la misma cantidad de datos. La salida de la capa GRU se conecta a la capa densa, la cual posee solo una neurona que se encarga de hacer la regresión del valor del ángulo a predecir, donde se utiliza una capa de activación lineal. Para entrenamientos, se incorpora una capa de pérdida con las métricas MSE o MAE y se utiliza el optimizador Adam. La arquitectura final de la red se muestra en la figura 3. 

![Arquitectura RNAM](https://user-images.githubusercontent.com/79665536/202080510-57b35954-6ed4-4b93-a0fa-95b68014990c.svg)
<p align = center>
Figura 3. Arquitectura neuronal de la RNAM.

<p align = justify>
En el entrenamiento se varía los hiperparámetros de neuronas de las capa GRU y el tamaño de lote del entrenamiento, cuyo resultado refleja que los valores óptimos son 32 y 16 respectivamente. Además de ello, se prueba la eficacia de utilizar una o dos capas GRU y se hace un preentrenamiento de los modelos, lo que corresponde a disminuir los errores de predicción al variar las métricas e inicializar los pesos con lo de una red previamente entrenada. Para el uso de las RNAM se necesita únicamente de ejecutar su archivo. Estos requieren que se especifique parámetros para su ejecución, los cuales son:

<p align = justify>
<li>project_name, el cual corresponde al nombre de proyecto debido a que se hace uso de la aplicación Weight and Biases para registrar los resultados de los entrenamientos de los modelos.</li>
<li>units, que corresponde a la cantidad de neuronas que se utilizan en la capa GRU de la RNAM bajo el escenario sintético como real.</li>
<li>epochs, corresponden a la cantidad de épocas a entrenar la RNAM.</li>
<li>batch_size, corresponde al tamaño de lote a utilizar en el entrenamiento de la RNAM.</li>
<li>loss_name, corresponde al nombre del gráfico de pérdida.</li>
<li>predict_name, corresponde al nombre del gráfico de predicción.</li>
<li>model_namel, corresponde al nombre que le va a dar al modelo entrenado.</li>
<li>loss_type, corresponde a si se usa la métrica MSE o MAE en la función de pérdida de la RNAM.</li>
<li>load_model, se indica el nombre del modelo que se quiere cargar.</li>

<p align = justify>
La implementación de la RNAM bajo el escenario sintético presenta una respuesta satisfactoria con un error absoluto medio de $0.0621^{\circ}$ en sus predicciones. Paralelamente, se determina que el uso de una capa GRU es el adecuado para aprender el comportamiento del modelo, ya que el uso de dos capas GRU disminuye el error en $0.044^{\circ}$. La predición del model óptimo se muestra en la figura 4.

![Predict_perfecto](https://user-images.githubusercontent.com/79665536/203123184-102de1fa-595c-40cc-aced-74eebb569a51.svg)
<p align = center>
Figura 4. Predicción de la mejor RNAM sintética. El modelo posee solo una capa GRU y fue preentrenado dos veces.
  
<p align = justify>
Por otro lado, se implementa la RNAM con datos del PAHM físico, cuyos resultados también son satisfactorios prediciendo con un error absoluto medio de $1,0875^{\circ}$. Donde a diferencia de la RNAM sintética, el uso de dos capas GRU resulta eficaz, ya que reduce el error en  $2,817^{\circ}$. La predición del model óptimo se muestra en la figura 5.

![Predict_RNAM_17](https://user-images.githubusercontent.com/79665536/203123067-b49ae325-1639-40d0-82ec-f7bb719efb8b.svg)
<p align = center>
Figura 5. Predicción el modelo óptimo de la RNAM real. Em modelo posee 2 capas GRU y preentrenamiento
   
<p align = justify>
De los resultados obtenidos se puede concluir que el preentrenamiento de la red es necesario ya que reduce el MAE en las predicciones del modelo. En cuanto a la cantidad de capas GRU, para el escenario sintético es suficiente el modelado con una capa GRU, ya que el uso de dos capas GRU incrementa el tiempo de entrenamiento por una reducción del MAE de $0,044^{\circ}$ en comparación al resultado dado por el modelo entrenado con una capa GRU, por lo que se descarta. Por otro lado, para el escenario real, el uso de dos capas GRU proporciona una mayor reducción del error comparado con el modelo de una capa GRU.

<p align = justify>
En cuanto a su comparación con el modelo sintético, este presenta un error absoluto promedio de $17,5177^{\circ}$ con respecto a la respuesta real de la PAHM, como se observa en la figura 3.

<p align = justify>
Por otro lado, el modelo matemático lineal permite aproximar el comportamiento dinámico del PAHM solo para ángulos pequeños, no obstante, en ángulos mayores a $14^{\circ}$ el error en las predicciones incrementa, lo que limita la aproximación del comportamiento del PAHM, lo que se muestra en la figura 6. En cambio, el uso de la RNAM permite una aproximación del comportamiento del PAHM en todo su espacio de operación, como se observa en la figura 5. Esto es conveniente para el uso de este como un simulador de la planta. Además, a pesar del efecto de las no linealidades y variaciones en el tiempo del PAHM, logra aprender ese comportamiento dinámico y dar una respuesta aproximada.

![modelo_vs_real](https://user-images.githubusercontent.com/79665536/203122959-79dd9894-d6d6-4cc6-9534-fcb04c11b6cf.svg)
<p align = center>
Figura 6. Comparación de la respuesta del modelo matemático con la PAHM.

# Redes neuronales y aprendizaje profundo

## Principales hitos históricos

### Fundaciones tempranas (1943–2011)

- 1943: McCulloch y Pitts proponen el primer modelo matemático de una neurona.
- 1958: Frank Rosenblatt desarrolla el **Perceptron**, considerado la primera red neuronal implementada.
- 1969: Minsky y Papert publican "Perceptrons", demostrando las limitaciones del simple perceptrón, que llevó al "invierno de la IA".
- 1986: Hinton, Rumelhart y Williams publican el algoritmo ***backpropagation***, que ** permite entrenar redes neuronales multicapa**.
- 1988: LeCun et al. presentan el primer modelo de red neuronal convocional para el reconocimiento de caracteres manuscritos (MNIST).
- 1997: Hochreiter & Schmidhuber presenta **LSTMs** (Long Short-Term Memory), fundamental para procesar secuencias y series temporales.

### The AI Boom (2012-present)

- Causas:
    - Aumento de la potencia de computación (GPU, TPU)
    - Disponibilidad de conjuntos de datos enormes (Internet, Big Data)
    - Avances en algoritmos y arquitecturas de red
    - Aumento considerable de la financiación y la inversión industrial

- Principales hitos:
    - 2012: **AlexNet** (Krizhevsky, Sutskever e Hinton) reduce el error en **ImageNet** a 15.3% (del 26%), demostrando el poder de las CNN y marcando el comienzo del boom.
    - 2014: Facebook **DeepFace** logra una precisión casi humana (97.35%) en reconocimiento facial.
    - 2014: Ian Goodfellow presenta **GANs** (Generative Adversarial Networks), revolucionando la generación de contenidos.
    - 2015: **ResNet** de Microsoft Research introduce conexiones residuales, permitiendo la formación de redes con más de 100 capas.
    - 2016: DeepMind **AlphaGo** derrota al campeón mundial Lee Sedol usando redes neuronales profundas entrenadas con el aprendizaje supervisado y reforzado con técnicas avanzadas de búsqueda de árboles de Monte Carlo. ([Documental](https://www.youtube.com/watch?v=WXuK6gekU1Y)).
    - 2017: La arquitectura **Transformer** aparece con la publicación de "**Atención es todo lo que necesitas**" de Google Brain, transformando el procesamiento del lenguaje.
    - 2018: El **BERT** de Google establece nuevos registros en la comprensión del lenguaje natural.
    - 2020: OpenAI **GPT-3** demuestra capacidades emergentes en modelos de lenguaje a gran escala.
    - 2021: **Modelos de difusión** (DALL-E, GLIDE) comienzan a dominar la generación de imagen realista.
    - 2022: OpenAI **ChatGPT** (GPT-3.5) populariza asistentes de conversación.
    - 2023: OpenAI **GPT-4o** y la proliferación de **multimodal** modelos (texto, imagen, audio, vídeo).
    - 2024: Primeros modelos con capacidades avanzadas de razonamiento (**OpenAI o1**).
    - 2025: **DeepSeek-R1** (*pesos abiertos*) reduce el costo de los modelos de idiomas con un rendimiento similar al o1.

## The Perceptron: The Artificial Neuron

El perceptrón es la unidad fundamental de una red neuronal, inspirada en el funcionamiento básico de una neurona biológica. Es la forma más simple de una red neuronal, que de muchas maneras puso las bases para redes más complejas.

> [¿Qué es una Red Neural? Parte 1: La neurona tóxica DotCSV](https://www.youtube.com/watch?v=MRIv2IwFTPg&list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0&index=7)

> [ChatGPT se hace de 100 millones de estos [El Perceptrón] https://www.youtube.com/watch?v=l-9ALe3U-Fg)

### Estructura del Perceptrón

[XPHXIMG0XPHX](https://aiml.com/what-is-a-perceptron/)

Un perceptron toma varias entradas binarias y produce una única salida binaria. Funciona pesando la importancia de cada entrada y, si la suma de los insumos ponderados supera un determinado umbral, "fuego" (salida a 1). De lo contrario, no lo hace (salida 0).

Un perceptrón consiste en:
1.  ** Entradas**: Valores numéricos que representan características de los datos.
2.  **Pesas**: Valores que determinan la importancia de cada entrada. Un peso superior significa que la entrada tiene más influencia en la salida.
3.  **Bias**: Un parámetro adicional que permite ajustar el umbral de activación, haciendo que el modelo sea más flexible.
4.  **Sum de peso**: Las entradas se multiplican por sus pesos correspondientes y se resumen.
5.  ** Función de acción**: Esta función decide si la neurona debe activarse o no sobre la base de la suma ponderada. En un perceptrón simple, esta es una simple función paso.

### Limitaciones del Perceptrón Simple

El perceptrón simple sólo puede resolver problemas linealmente separables. Esto significa que puede clasificar datos que pueden ser separados por una sola línea recta. Puede aprender operaciones lógicas simples como AND y OR, pero famosamente falla con la operación XOR, que no es linealmente separable. Esta limitación llevó a un período de menor interés en las redes neuronales, conocido como el "invierno AI"

## Multilayer Neural Networks (MLP)

Para superar las limitaciones del perceptrón simple, los investigadores desarrollaron redes neuronales multicapas, también conocidas como Perceptrones Multilayer (MLPs). Estas redes consisten en múltiples capas de perceptrones, permitiéndoles aprender patrones mucho más complejos.

> [3Blue1Brown - ¿Pero qué es una red neuronal?](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Estructura de una red neuronal multicapa

Una red neuronural multicapa contiene:
1.  **La capa de entrada**: Recibe los datos iniciales.
2.  **Las capas ocultas**: Una o más capas que realizan transformaciones internas en los datos. Estas son las claves para aprender relaciones complejas y no lineales.
3.  **La capa de salida**: Produce el resultado final (por ejemplo, una clasificación o un valor de regresión).


![MLP](./img/mlp.gif)

### ¿Por qué necesitamos capas ocultas?

Las capas ocultas permiten que la red aprenda características jerárquicas. Cada capa aprende a reconocer ciertos patrones en los datos, y capas posteriores combinan estos patrones para aprender representaciones aún más complejas. Esto es lo que da el aprendizaje profundo su poder.

## Funciones de activación

Las funciones de activación son un componente crucial de las redes neuronales. Presentan la no-linearidad en el modelo, que es lo que permite a la red aprender de datos que no es linealmente separable. Sin ellos, una red neuronal sería sólo un modelo lineal, sin importar cuántas capas tenga.

### Funciones de activación principal

1.  **Sigmoid**: Comprende cualquier entrada en un rango entre 0 y 1. A menudo se utiliza en la capa de salida para problemas de clasificación binaria.
2.  ** Tangente Hiperbólico (tanh)**: Similar a la sigmoide, pero comprime valores en un rango entre -1 y 1.
3.  **ReLU (Rectified Linear Unit)**: Esta es una de las funciones de activación más populares en el aprendizaje profundo moderno. Es muy simple: produce la entrada si es positiva, y 0 de lo contrario. Es computacionalmente eficiente y ayuda a mitigar el problema "desvanecedor gradiente".
4.  **Softmax**: A menudo se utiliza en la capa de salida para problemas de clasificación de varias clases. Convierte un vector de puntajes crudos en una distribución de probabilidad, donde cada valor representa la probabilidad de la entrada perteneciente a una determinada clase.

## Formación: **Descenso gradiente**

> [¿Qué es Gradient Descent? Algoritmo de Inteligencia Artificial ← DotCSV](https://www.youtube.com/watch?v=A6FiCDoz8_4)

> [3Blue1Brown - Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)

Entrenar una red neuronal es el proceso de encontrar los pesos y sesgos óptimos que permiten a la red hacer predicciones precisas. Esto se hace minimizando una función **pérdida**, que mide hasta qué punto las predicciones del modelo son de los valores reales correctos.

El algoritmo de optimización más común utilizado para entrenar redes neuronales es **Gradient Descent**. Funciona ajustando iterativamente los parámetros del modelo (pesos y sesgos) en la dirección que reduce la pérdida.

1.  **Conceptos clave**:
    -   **Gradient**: Un vector que apunta en la dirección del ascenso más pronunciado de la función de pérdida. Para minimizar la pérdida, nos movemos en la dirección opuesta del gradiente.
    -   ** Tasa de aprendizaje**: Un pequeño valor que controla el tamaño de los pasos que tomamos durante la optimización. Una tasa de aprendizaje demasiado alta puede hacer que el modelo supere la solución óptima, mientras que una que es demasiado baja puede hacer que el proceso de entrenamiento sea muy lento.

2.  **Proceso**:
    -   El modelo hace una predicción (esto se llama el *paso adelante*).
    -   La pérdida se calcula comparando la predicción con el valor real.
    -   Se calcula el gradiente de la pérdida con respecto a cada parámetro.
    -   Los parámetros se actualizan dando un pequeño paso en la dirección opuesta del gradiente.

3.  **Variantes**:
    -   **Descenso de ingredientes* Utiliza todo el conjunto de datos para calcular el gradiente a cada paso. Es lento y intensivo en memoria para grandes conjuntos de datos.
    -   **Descenso de gradiente estocástico**: Utiliza un único punto de datos seleccionado al azar para calcular el gradiente a cada paso. Es mucho más rápido pero puede ser ruidoso.
    -   ***Mini-batch Gradient Descent***: Un compromiso entre los dos. Utiliza un pequeño y aleatorio lote de datos para calcular el gradiente. Este es el enfoque más común en el aprendizaje profundo.

## ***Backpropagation***

La retropropagación es el algoritmo que nos permite capacitar eficientemente redes neuronales profundas y multicapas. Es el motor que potencia el aprendizaje profundo moderno.

> [¿Qué es una Red Neural? Parte 3: Backpropagation ← DotCSV](https://www.youtube.com/watch?v=eNIqz_noix8&list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0&index=10)

> [3Blue1Brown - Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)

### ¿Cómo funciona?

1.  **Pase adelante**: La red calcula su salida para una entrada dada, pasando los datos de la capa de entrada a través de las capas ocultas a la capa de salida.
2.  **Error Calculation**: La salida se compara con el valor correcto para calcular el error o la pérdida.
3.  **Paso de regreso**: El error se propaga hacia atrás a través de la red, desde la capa de salida a la capa de entrada. Durante este proceso, el algoritmo calcula cuánto cada peso y sesgo contribuyó al error general.
4.  ** Actualización de peso**: Los pesos y sesgos se ajustan en función de su contribución al error, utilizando un algoritmo de optimización como Gradient Descent.

La base matemática de la retropropagación es la regla ** cadena** del cálculo, que le permite computar eficientemente los gradientes para todos los parámetros de la red.

## ***Deep Learning***

*Aprendizaje profundo* se refiere a redes neuronales con muchas capas ocultas (de ahí "deep"). Estas arquitecturas profundas permiten a los modelos aprender representaciones de datos en varios niveles de abstracción.

![Aprendizaje profundo](./img/deep_learning.jpg)

### Ventajas del aprendizaje profundo

1.  **Aprendizaje automático de características**: A diferencia del aprendizaje tradicional de la máquina, donde la ingeniería de características es a menudo un proceso manual y prolongado, los modelos de aprendizaje profundo pueden aprender las características relevantes directamente de los datos.
2.  ** Capacidad para modelar relaciones complejas**: El aprendizaje profundo se destaca en las relaciones complejas y no lineales, lo que hace que sea particularmente poderoso para datos no estructurados como imágenes, texto y sonido.
3.  **Transferibilidad**: El conocimiento aprendido de una tarea puede ser transferido a otra. Esto se conoce como **transfer learning**, y nos permite construir modelos poderosos incluso con datos limitados.

### Popular Deep Learning Architectures

1.  ** Redes neuronales revolucionarias**: Especializado para el procesamiento de datos similares a la red, como imágenes. Utilizan capas convolutivas para detectar patrones espaciales como bordes, texturas y formas.
2.  **Recurrent Neural Networks (RNNs) and LSTM/GRU**: Diseñado para datos secuenciales, como series temporales o texto. Tienen una forma de "memoria" que les permite mantener información sobre entradas anteriores en la secuencia.
3.  **Transformers**: Una arquitectura más moderna, también diseñada para datos secuenciales, que ha revolucionado el procesamiento del lenguaje natural. Utiliza un mecanismo llamado "atención" para pesar la importancia de diferentes partes de los datos de entrada. Modelos como BERT y GPT se basan en la arquitectura Transformer.
4.  ** Redes adversarias de carácter consultivo (GAN)**: Consista de dos redes, un generador y un discriminador, que compiten entre sí. Se utilizan para generar nuevos datos sintéticos similares a los datos de entrenamiento (por ejemplo, creando imágenes realistas).
5.  **Autoencoders**: Se utiliza para el aprendizaje no supervisado, normalmente para la reducción de la dimensionalidad o la detección de anomalías. Aprenden a comprimir datos en una representación más pequeña y luego lo reconstruyen de nuevo a su forma original.

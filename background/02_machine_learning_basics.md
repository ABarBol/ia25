# Paradigmas de aprendizaje automático

El aprendizaje automático permite a los sistemas aprender de datos, identificar patrones y tomar decisiones con mínima intervención humana. A diferencia de la programación tradicional, donde los programas están diseñados con reglas explícitas, ML construye modelos aprendiendo de datos.

> [Modelos para entender una realidad caótica](https://www.youtube.com/watch?v=Sb8XVheowVQ)

Un **modelo** es una descripción abstracta y articulada de una realidad. En el aprendizaje automático, los modelos son lo que **entren** con datos usando un algoritmo de aprendizaje. El modelo aprende a ajustarse a un gran número de ejemplos y luego se utiliza para predecir la respuesta correcta para nuevos datos de entrada no vistos.

La forma en que un modelo aprende se define por su paradigma de aprendizaje, que se clasifica según el tipo de supervisión o retroalimentación que recibe durante el entrenamiento.

![ML paradigms](./img/ml_paradigms.jpg)

> [¿Qué es el aprendizaje supervisado y no supervisado? Silencio](https://www.youtube.com/watch?v=oT3arRRB2Cw)

<img src="img/tipos-ml-clasico.jpg" width="400">

## Enseñanza supervisada

En el aprendizaje supervisado, el conjunto de datos de entrenamiento es **marcado con la respuesta correcta**. El algoritmo de aprendizaje recibe un conjunto de datos de entrenamiento y, sabiendo la respuesta correcta para cada ejemplo, infiere un modelo que genera esa respuesta.

<img src="img/reglas_vs_ml.jpg" width="500">

### Clasificación

Una tarea típica de aprendizaje supervisada es **clasificación**. Considere un filtro de spam: un algoritmo puede aprender examinando muchos ejemplos de correos electrónicos ya etiquetados como "spam" o "no spam". Puede inferir que ciertas palabras están casi siempre asociadas con el spam, mientras que los correos electrónicos de destinatarios específicos nunca son spam. Los ejemplos más etiquetados en los que se entrena el algoritmo, mejor se convierte en el filtro de spam.

Otro ejemplo es el reconocimiento manual de dígitos, donde el algoritmo recibe imágenes de dígitos y debe clasificarlos de 0 a 9.

Tipos de clasificación:
- **Binario**: La salida tiene dos clases (por ejemplo, spam/no spam, positivo/negativo).
- **Multiclass**: La salida tiene más de dos clases (por ejemplo, reconocimiento digital, clasificación de imágenes).
- *Multilabel* Una instancia puede ser asignada múltiples etiquetas (por ejemplo, etiquetando música con géneros).

### Regreso

En ** problemas de regresión**, el objetivo es predecir un **valor continuo**. Por ejemplo, predecir el precio de una casa basado en sus características (número de habitaciones, tamaño del jardín, ubicación, etc.). En este caso, en lugar de una etiqueta de clase, cada ejemplo se etiqueta con un valor numérico (el precio de la casa).

## Aprendizaje no supervisado

En el aprendizaje no supervisado, el conjunto de datos de entrenamiento es **no etiquetado**. El objetivo es descubrir patrones ocultos o estructuras intrínsecas en los datos.

[XPHXIMG0XPHX](https://www.linkedin.com/pulse/supervised-vs-unsupervised-learning-whats-difference-smriti-saini)

[XPHXTAG0XPHX](https://www.researchgate.net/figure/Supervised-and-unsupervised-machine-learning-a-Schematic-representation-of-an_fig3_351953193)

### Clustering

**Clustering** implica agrupar puntos de datos en grupos basados en la similitud. Un ejemplo es la segmentación del cliente, donde los clientes se agrupan en segmentos similares a productos y servicios más adaptados. Esto se utiliza en sistemas de recomendación, marketing y más.

### Association Rule Mining

Mientras grupos de agrupación instancias (como clientes), **la política de asociación minera** descubre relaciones entre los artículos. Por ejemplo, identificar que los clientes que compran pan también tienden a comprar mantequilla ("análisis de la cesta del mercado").

### Reducción de la dimensión

**Reducción de la dimensión** tiene como objetivo reducir el número de variables (características) en un conjunto de datos. Esto es útil cuando se trata de datos de alta dimensión donde algunas características pueden ser redundantes o irrelevantes. Puede reducir el tiempo de entrenamiento y mejorar la precisión del modelo. Una técnica común es **Análisis del componente principal (PCA)**.

### Detección de anomalías

Esta tarea se centra en identificar puntos o patrones de datos inusuales que se desvían significativamente de la norma. Se utiliza en detección de fraudes, monitoreo de sistemas y seguridad para marcar datos que podrían indicar errores o ataques.

## Reinforcement Learning

En ** aprendizaje de refuerzo**, un **agente** aprende interactuando con un **ambiente**. El agente realiza acciones, y el medio ambiente proporciona comentarios en forma de **rewards** o **penalties**. El objetivo del agente es aprender una **política** (una estrategia para elegir acciones) que maximice su recompensa acumulativa con el tiempo.

Este paradigma se encuentra en robótica (por ejemplo, enseñar a un robot a caminar) y juegos (por ejemplo, entrenar una AI para jugar ajedrez o Go).

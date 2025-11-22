# Curso de Aprendizaje IA y Máquina

-..

## *CONTENIDOS*

### **Unit 1: Foundations of Artificial Intelligence and Business Impact**
* **Objetivo:** Establecer una base conceptual sólida, entender el "por qué" de la IA y sus implicaciones.
* **Contenidos:**
    * Principios de sistemas inteligentes: ¿Qué es AI?
    * Strong AI vs. Weak AI: Caracterización, usos y posibilidades.
    * AI vs Machine Learning vs Deep Learning: Definiciones y relaciones.
    * Data Science and Big Data: The role of data in AI.
    * Ciclo de vida AI: De la recopilación de datos a la implementación y monitoreo modelo.
    * Aprendizaje supervisado, no supervisado y reforzado: Conceptos básicos y diferencias (respecto de alto nivel).
    * Campos de aplicación AI: Identificación de sectores y casos de uso.
    * Convergencia Tecnológica: Cómo AI, IoT, Cloud y Blockchain unifican los procesos y mejoran la toma de decisiones estratégicas.
    * AI en negocios: Mejora de la eficiencia operativa, nuevas interacciones y modelos de negocio.

-..

### **Unit 2: Ecosistema de programación para AI**
* **Objetivo:** Adquire essential programming tools for AI development.
* **Contenidos:**
    * Introducción a Python: El lenguaje estándar de la industria para AI.
    * Otros idiomas: R, Julia y sus aplicaciones específicas.
    * Ambientes con uv: Gestión de dependencias y entornos virtuales.
    * Medios de desarrollo: Jupyter Notebooks, Google Colab, VS Code.
    * Configuración: VSCode con Python, Jupyter y Git. Medio ambientes y gestión de dependencia con uv.
    * **Programación con agentes de IA:**
        * **GitHub Copilot in VS Code:** Compleción y generación de código impulsado por AI.
        * Moderno flujo de trabajo de desarrollo: Utilizando AI como socio de programación de pares.
        * Mejores prácticas: Ingeniería avanzada para la generación de códigos, revisión de sugerencias de IA.
        * IA-asistida depuración y refactorización.
        * Aprovechando agentes de inteligencia artificial durante todo el ciclo de vida del desarrollo.
        * **Nota:** Todos los proyectos del curso se desarrollarán utilizando modernos flujos de trabajo asistidos por AI.
    * Control de versiones con Git y GitHub: Colaboración y gestión de proyectos.
    * Bibliotecas fundamentales:
        * **NumPy:** Para computación numérica.
        * **Pandas:** Para la manipulación y análisis de datos.

-..


### **Proyecto 1: Práctica: Crear un repositorio GitHub con un juego tic-tac-toe. La máquina debe ser capaz de jugar contra un humano usando un algoritmo mini-max.**
    - Mejoras:
        - Implementar una interfaz gráfica de usuario (GUI) usando bibliotecas como Tkinter o Pygame.
        - Añadir una opción para niveles difíciles: el agente juega al azar, el agente utiliza el algoritmo minimax.
        - Añadir soporte para diferentes niveles de dificultad limitando la profundidad de la búsqueda minimax.
        - Añade un sistema de puntuación para rastrear victorias, pérdidas y sorteos.
        - Incluye pruebas unitarias para garantizar la corrección de la lógica del juego y las decisiones AI.
--

### **Proyecto 2: Procesando MNIST con NumPy**

-..

### **Unit 3: The Three Machine Learning Paradigms - A Practical Introduction**
* **Objetivo:** Entender los tres enfoques fundamentales del aprendizaje automático a través de ejemplos prácticos y preparar a los estudiantes para inmersiones más profundas en cada paradigma.
* **Duración*
* **Contenidos:**
    * **Overview: The Three Learning Paradigms**
        * ¿Qué problemas resuelve cada paradigma?
        * ¿Cuándo utilizar el aprendizaje de refuerzo supervisado vs no supervisado?
        * Aplicaciones del mundo real de cada enfoque.
        * Las señales de aprendizaje: etiquetas, patrones y recompensas.
    * **Paradigma 1: Aprendizaje supervisado - Aprendizaje de Ejemplos**
        * Concepto: Aprender una asignación de insumos a salidas utilizando datos etiquetados.
        * ** Ejemplo simple:** K-Nearest Neighbors (KNN) en el conjunto de datos de Iris.
        * Manos a mano: Cargar modelo de scikit-learn pre-entrenado, hacer predicciones, evaluar la precisión.
        * Información clave: El modelo aprende de ejemplos con respuestas conocidas.
        * Teaser: "En la Unidad 4, nos sumergimos profundamente en muchos algoritmos supervisados."
    * **Paradigma 2: Aprendizaje no supervisado - Encontrar patrones ocultos**
        * Concepto: Descubrir la estructura en datos sin etiquetas.
        * ** Ejemplo:** Medios K agrupando datos sintéticos 2D.
        * Manos a mano: Generar datos, aplicar agrupaciones, visualizar resultados.
        * Información clave: El modelo agrupa puntos de datos similares automáticamente.
        * Teaser: "En la Unidad 5, exploraremos PCA, DBSCAN y detección de anomalías".
    * **Paradigm 3: Reinforcement Learning - Learning through Trial and Error**
        * Concepto: Un agente aprende interactuando con un ambiente y recibiendo recompensas.
        * Componentes básicos:
            * **Agente:** El alumno/aprendizaje de la decisión.
            * ** Medio ambiente** El mundo con el que el agente interactúa.
            * **Estado:** Situación actual del agente.
            * **Acción** Opciones disponibles para el agente.
            * **Retroceder:** Indicación de retroalimentación (positiva o negativa).
        * **Q-Learning Algorithm:**
            * Q-table: Guardar recompensas esperadas para parejas de acción del estado.
            * Exploration vs Exploitation: Balancing trying new actions vs using known good ones.
            * Proceso de aprendizaje: Actualización de valores Q basados en la experiencia.
            * Implementación sencilla con diccionarios y bucles.
        * ** Ejemplos prácticos:**
            * GridWorld: Navigate una cuadrícula para alcanzar una meta evitando obstáculos.
            * Tic-Tac-Toe: Aprenda los movimientos óptimos a través del auto-juego.
        * Información clave: Nadie le dice al agente qué hacer; aprende de las consecuencias.
    * **Comparando los paradigmas:**
        * Tabla de comparación: Requisitos de datos, tipo de retroalimentación, aplicaciones comunes.
        * Discusión: Por qué diferentes problemas necesitan diferentes enfoques.
    * **Python Programming Practice:**
        * Programación orientada hacia objetos: Clases para agentes, ambientes, juegos.
        * Estructuras de datos: Diccionarios para Q-tables, listas de representación estatal.
        * Flujo de control: Opciones para los episodios de entrenamiento, condiciones para la toma de decisiones.
        * Organización del código: Separar la lógica del juego, la lógica del agente y el código de entrenamiento.
        * Pruebas: Pruebas de unidad para reglas de juego y comportamiento de agente.
    * **GitHub Workflow:**
        * Estrategias de ramificación para el desarrollo de características.
        * Significativo comprometer mensajes e historia de la versión.
        * Documentación de código y archivos README.
        * Prácticas colaborativas de codificación.

-..

### **Proyecto 3: Reinforcement Learning Game Agent**
*Aplicado después de la Unidad 3 - Paradigmas ML*
* **Objetivo:** Implementar un agente de Q-learning para dominar un juego simple, practicando Python OOP y GitHub workflow.
* ** Componentes clave:**
    * **Entorno del juego** Elige uno de los tres niveles de dificultad:
        * **Nivel 1 (Principiante):** Navegación de GridWorld (5×5 grid, obstacles, goal).
        * **Nivel 2 (Intermedio):** Tic-Tac-Toe contra el oponente aleatorio.
        * **Nivel 3 (Advanced):** Tic-Tac-Toe contra el oponente del algoritmo Minimax.
    * ** Aplicación del aprendizaje en la Sede:**
        * Representación del Estado: Cómo codificar situaciones de juego.
        * Espacio de acción: Los movimientos válidos de cada estado.
        * Diseño de recompensa: +1 para ganar, -1 para la pérdida, -0.1 para cada movimiento, etc.
        * Actualización Q-table: Implementación de la fórmula Q-learning.
        * Hyperparameters: Tasa de aprendizaje (α), factor de descuento (γ), tasa de exploración (ε).
    * **Proceso de formación:**
        * Lazo de entrenamiento: Jugando muchos episodios para mejorar.
        * Decaimiento de Epsilon: Reducción gradual de la exploración con el tiempo.
        * Seguimiento de rendimiento: Velocidad de ganancia, recompensa promedio, curvas de entrenamiento.
    * **Evaluación**
        * Prueba contra el oponente al azar.
        * Prueba contra estrategias basadas en normas.
        * Visualizar los valores de Q aprendidos.
    * **Requisitos de calidad del proyecto:**
        * Estructura adecuada de clase (clases de Agente, Medio Ambiente, Juego).
        * Documentación: Documentos, comentarios, README.
        * Pruebas de unidad: Prueba reglas del juego, movimiento de validación, actualizaciones de Q-table.
        * Git workflow: Multiple commits with clear messages, branching if needed.
* ** Resultados de aprendizaje:**
    * Comprender los conceptos de RL mediante la implementación.
    * Practicar Python programación y diseño OOP.
    * Desarrollar hábitos de codificación profesionales (testing, documentación, control de versiones).
    * Construir confianza antes de abordar complejos algoritmos de aprendizaje supervisados.
* **Nota:** Se trata de un proyecto RL simplificado y educativo centrado en los fundamentos y la práctica de codificación, no en los sistemas RL de nivel de producción.

-..

### **Unit 4: Aprendizaje de la máquina supervisada - Dive profunda**
* **Objetivo:** Domine los algoritmos y técnicas para aprender de datos etiquetados para hacer predicciones precisas.
* **Contenidos:**
    * Principios de aprendizaje automático: Datos, patrones y predicciones.
    * EDA y Preprocesamiento de datos: Limpieza, normalización y selección de características.
    * **Regreso** Predicción de valores continuos.
    * ** Clasificación:** Predicción de categorías.
    * Algoritmos clave:
        * K-Nearest Neighbors (KNN)
        * Regreso lineal
        * Regreso logístico
        * Árboles de decisión
        * Soporte Vector Machines (SVM)
        * Métodos de conjunto: Embalaje, Boosting y Stacking.
        * Bosques aleatorios
    * Utilizando **Scikit-Learn** para implementar estos modelos.
    * Evaluación modelo:
        * **Para regresión:** MAE, MSE, RMSE, R-squared y análisis residual.
        * **Para la clasificación** Precisión, Precisión, Recordar, F1-Score, matriz de confusión, ROC-AUC.
        * ** Técnicas generales:** Dividencias cruzadas, tren/validación/test.
    * Optimización y ajuste del hiperparametro.

-..


### **Proyecto 4: Sistema de Predicción de Precios Inmobiliarias en Miami Housing Dataset**
* Aplicado después de la Unidad 4 - Aprendizaje supervisado*
* **Objetivo:** Proyecto de regresión completo utilizando datos inmobiliarios para predecir precios de propiedad.
* ** Componentes clave:**
    * **Adquisición de datos:** Conjuntos de datos inmobiliarios con múltiples características (ubicación, tamaño, servicios, etc.).
    * **Ingeniería de la industria:** Crear nuevas variables, manejar datos categóricos, características geográficas.
    * ** Comparación modelo:** regresión lineal, Bosque aleatorio, XGBoost, regresión SVM.
    * **Evaluación avanzada** Validación cruzada, curvas de aprendizaje, análisis residual.
    * **Aplicación de negocios:** Herramienta de estimación de precios con intervalos de confianza.
* ** Mejora de la red neuronal** Implementar la regresión MLP y comparar con los modelos ML tradicionales.


-..

### **Unit 5: Aprendizaje sin supervisión de la máquina - Dive profunda**
* **Objetivo:** Técnicas maestras para descubrir patrones ocultos y estructuras en datos no etiquetados.
* **Contenidos:**
    * **Clustering:** Grouping similar data points.
        * K-medios, DBSCAN y algoritmos de agrupación jerárquica.
    * ** Reducción de la dimensión:** simplificar los datos complejos.
        * **Análisis principal de componentes (PCA):** Extracción y visualización de objetos.
        * t-SNE para la visualización de datos de alta dimensión.
        * Aplicaciones en ingeniería de características y reducción de ruido.
    * **Detección de anomalías:** Identificación de atípicos y patrones inusuales.
    * métricas de evaluación para el aprendizaje no supervisado.

-..



### **Proyecto 5: Sistema de detección de intrusiones (IDS) con enfoque híbrido ML**
* Aplicado después de la Unidad 5 - Aprendizaje no supervisado*
* **Objetivo:** Proyecto integral de ciberseguridad utilizando técnicas supervisadas y no supervisadas.
* ** Componentes clave:**
    * **Real datasets:** NSL-KDD o CICIDS2017 para ataques de red reales.
    * **Página 1 - Sin supervisión:**
        * PCA para reducción de dimensionalidad y visualización de características.
        * Detección de agrupación/análisis para establecer la base de conducta normal.
        * DBSCAN e Isolation Forest para la detección de anomalías.
    * **Phase 2 - Supervisado:** Clasificación de ataques conocidos con alta precisión.
    * **Ensemble approach:** Combinando ambos enfoques para maximizar la detección y minimizar falsos positivos.
    * **Métricas especializadas:** Precisión equilibrada/Recaimiento para minimizar falsos positivos.
    * ** Aplicación práctica:** Dashboard en tiempo real para el monitoreo de redes.
* ** Mejora de la red neuronal** Autoencoders profundos para detección de anomalías, CNNs para reconocimiento de patrones de tráfico de red.

-..

## 1ST TERM EVALUATION POINT

-..

### **Unidad 6: Fundamentos de aprendizaje profundo con PyTorch**
* **Objetivo:** Domine los cimientos del aprendizaje profundo y la implementación de redes neuronales desde cero usando PyTorch.
* **Contenidos:**
    * **Neural Network Foundations:**
        * De Perceptron a Perceptrones Multi-Layer (MLP).
        * Teorema de aproximación universal y profundidad de red.
        * Funciones de activación: ReLU, sigmoide, tanh y variantes modernas.
        * El algoritmo de retropropagación: comprensión del flujo de gradiente.
    * **Deep Learning with PyTorch:**
        * Introducción a **PyTorch** marco y operaciones de tensor.
        * Construcción de redes neuronales con `nn.Module`.
        * Proceso de entrenamiento: Funciones de pérdida, optimizadores (SGD, Adam) y retropropagación.
        * Técnicas de regularización: abandono, normalización de lotes, descomposición de peso.
        * Capacitación de las mejores prácticas: programación de la tasa de aprendizaje, parada temprana.
        * Debugging neural networks: gradient check, learning curves.
    * **Hands-on Implementation:**
        * Construyendo MLPs desde cero para clasificación y regresión.
        * Clasificación MNIST digital con redes de alimentación.
        * Ajuste del hiperparametro y evaluación del modelo.
        * Visualización de dinámicas de capacitación y límites de decisión.

-..

### **Proyecto 6: Neural Network Fundamentals - TensorFlow Playground Tutorial**
* Aplicado después de la Unidad 6 - Fundamentos de aprendizaje profundo*
* **Objetivo:** Cree un tutorial completo que cubra todos los escenarios en TensorFlow Playground para construir intuición sobre el comportamiento de la red neuronal.
* ** Componentes clave:**
    * **Exploración sistemática** Document conduct across all datasets and architectures.
    * ** Análisis de profundidad de red:** Compare redes superficiales vs profundas.
    * ** Funciones de acción** Experimento con diferentes funciones de activación.
    * ** Efectos de la tasa de aprendizaje:** Demostrar impacto en la convergencia.
    * **Regularización** Mostrar efectos de goteo y otras técnicas.
    * **Personalización:** Demostrar y explicar patrones de superposición.
* ** Resultados de aprendizaje:** Construir una fuerte intuición antes de implementar redes desde cero.

-..

### **Unit 7: Computer Vision with Convolutional Neural Networks**
* **Objetivo:** Arquitecturas Master CNN y técnicas modernas de visión informática utilizando modelos pre-entrenados y aprendizaje de transferencia.
* **Contenidos:**
    * ** Redes neuronales revolucionarias (CNN) - Profundidad:**
        * El reto de los datos de imagen: estructura espacial y eficiencia del parámetro.
        * Componentes de arquitectura CNN:
            * Capas convolutivas: filtros, mapas de características, campos receptivos.
            * Capas de estanqueidad: estanqueidad máxima, estanqueidad media.
            * capas totalmente conectadas y extracción de características.
        * Comprensión de las convoluciones: visualización de filtros y mapas de características.
        * Arquitecturas clásicas: LeNet, AlexNet, VGG, ResNet, EfficientNet.
    * **Transfer Learning and Pre-trained Models:**
        * Por qué transferir obras de aprendizaje: características aprendidas y adaptación de dominio.
        * Utilizando modelos pre-entrenados de torchvision y timm.
        * Estrategias de ajuste fino: capas de congelación, tasas de aprendizaje.
        * Extracción de características para tareas de aguas abajo.
    * ** Aplicaciones de Visión Informática Moderna:**
        * **Detección de objetos:** YOLO, arquitecturas R-CNN más rápidas.
        * **Segmentación semántica:** conceptos U-Net y FCN.
        * **Similitud de imagen:** Extracción de objetos y búsqueda de similitud.
        * **Inferencia de tiempo real:** Optimización para la producción.
    * **Aprendizaje sin supervisión:**
        * Arquitectura: estructura de encoder y cuello de botella.
        * Representación espacial latente y reducción de la dimensión.
        * Objetivos de capacitación: pérdida de reconstrucción.
        * Aplicaciones: denoización, compresión, aprendizaje de características, detección de anomalías.
        * Autoencoders Variational (VAE): enfoque probabilista al modelado generativo.
        * Comparación con PCA y otras técnicas de reducción de la dimensionalidad.
    * ** Herramientas prácticas y marcos:**
        * torchvision: datasets, modelos, transformas.
        * Transformadores Hugging Face para modelos de visión.
        * YOLO frameworks: ultralytics, YOLOv8.
* ** Ejemplos disponibles:**
    * **MNIST Ejemplos:**
        * `FNN_MNIST.ipynb` - Perceptron multicapa con técnicas de normalización y regularización
        * `CNN_MNIST.ipynb` - Arquitectura básica de la CNN para la clasificación de dígitos
        * `mnist-model-comparison.py` - Análisis comparativo entre regresión logística, NN de una sola capa y CNN
    * **CIFAR-10 Ejemplos:**
        * `CIFAR-10.ipynb`
        * `CIFAR10_gray.ipynb` - FNN vs CNN comparación en grayscale CIFAR-10
        * `models/baseline.py` y `models/mejorado.py` - Arquitecturas CNN básicas y mejoradas
    * **Key Learning Concepts:**
        * Técnicas de preprocesamiento y normalización de datos
        * Principios de diseño de arquitectura para FNN vs CNN
        * Aplicación y optimización de los circuitos de capacitación
        * Comparación de rendimiento y análisis de errores
        * Entendiendo por qué las CNNs sobresalen en tareas de imagen
* ** Resultados de aprendizaje:** Los estudiantes comprenderán los fundamentos de la red neuronal, verán las implementaciones prácticas y comprenderán las ventajas de las CNN para las tareas de visión informática.

-..

### **Proyecto 7B: Sistema Inteligente de Visión Informática**
*Aplicado después de la Unidad 7 - Computer Vision con CNN*
* **Objetivo:** Conductor completo de visión computarizada utilizando modelos pre-entrenados, detección de objetos y búsqueda de similitudes con bibliotecas de alto nivel.
* ** Componentes clave:**
    * **Aplicación de la OLO:** Detección de objetos en tiempo real en imágenes/video usando ultralitica o similar.
    * ** Extracción de características:** Utilizando CNNs pre-entrenados (ResNet, EfficientNet) para extraer características de imagen.
    * *Base de datos de los vehículos* Guardar características de imagen en ChromaDB para búsqueda de similitud.
    * ** Búsqueda de la similaridad:** Encontrar imágenes visualmente similares usando incrustaciones vectoriales.
    * **Aprendizaje de traslado:** Ajuste un modelo en un conjunto de datos personalizado.
    * ** Aplicación práctica:** Motor de búsqueda visual para comercio electrónico o gestión de contenidos.
* ** Mejoras avanzadas:** Integración con LLMs para descripción de imágenes y etiquetado automatizado.
* ** Resultados de aprendizaje:** Aplica herramientas modernas de visión informática para resolver problemas del mundo real utilizando API de alto nivel.
* **Nota:** Este proyecto se centra en el uso de herramientas de producción, sin implementar algoritmos desde cero.

-..

### **Unidad 8: Datos secuenciales y la revolución transformadora**
* **Objetivo:** Master arquitecturas especializadas para datos secuenciales y temporales, desde RNNs a Transformers modernos.
* **Contenidos:**
    * **El desafío de los datos secuenciales**
        * Características de las secuencias: longitud variable, dependencias temporales, contexto.
        * Aplicaciones: Procesamiento de Lenguas Naturales (NLP), series temporales, discurso, vídeo.
        * Por qué las redes neuronales tradicionales luchan con secuencias.
    * **Recurrent Neural Networks (RNNs):**
        * Arquitectura: estado oculto como mecanismo de memoria.
        * Secuencias de procesamiento: un elemento a la vez.
        * Training RNNs: Backpropagation Through Time (BPTT).
        * **RNN:**
            * LSTM (Long Short-Term Memory): puertas para la memoria selectiva.
            * GRU (Gated Recurrent Units): mecanismo de medición simplificado.
            * RNNs bidireccionales: contexto desde ambas direcciones.
        * *Limitaciones* Desapareciendo gradientes, cuello de botella de procesamiento secuencial, dificultad con dependencias de largo alcance.
    * ** Modelos de secuencia a secuencia (Seq2seq):**
        * Arquitectura encoder-decoder para entrada / salida de longitud variable.
        * Aplicaciones: traducción automática, resumen de texto, chatbots.
        * Limitaciones: falta de información, vector de contexto fijo.
    * **El Mecanismo de Atención - El avance:**
        * Motivación: permitiendo que el modelo se centre en las partes pertinentes de la entrada.
        * **Atención fundamental:**
            * Consulta, clave, paradigma de valor.
            * Marcas de atención y suma ponderada.
            * Suave atención vs atención dura.
        * Atención en seq2seq: contexto dinámico para cada paso decodificador.
    * **Arquitectura del transformador - "Atención es todo lo que necesitas":**
        * Eliminar la recurrencia: procesamiento paralelo de secuencias.
        * ** Mecanismo de autoatención:**
            * Relatando diferentes posiciones dentro de una sola secuencia.
            * Atención multicabezas: capturar diversas relaciones en paralelo.
            * Consideraciones de complejidad y eficiencia.
        * **Codificación profesional:** Información de orden de secuencia de inyección.
        * **Los bloques de construcción del transportador:**
            * ** Arquitectura de encoder:**
                * Autoatención para la comprensión del contexto bidireccional.
                * Redes de alimentación y conexiones residuales.
                * Normalización de capas.
                * Aplicaciones: BERT, RoBERTa para clasificación y comprensión.
            * **Arquitectura decodificador:**
                * Autoatención enmascarada para la generación autoregresiva.
                * Evita mirar las futuras fichas durante el entrenamiento.
                * Aplicaciones: Familia GPT para la generación de texto.
            * **Ecoder-Decoder architecture:**
                * Interatención: conectar salidas de encoder a decodificador.
                * Aplicaciones: T5, BART para la traducción, resummarización, tareas seq2seq.
        * **Transformers vs RNNs:** Ventajas en la paralización, dependencias de largo alcance y eficiencia.
    * **Hugging Face Ecosystem - Transformers prácticos:**
        * Introducción a las bibliotecas **Hugging Face**:
            * `transformers`: Modelos pre-entrenados para diversas tareas.
            * `datasets`: Acceso a miles de conjuntos de datos.
            * `tokenizers`: Procesamiento y preparación de texto rápido.
        * Trabajando con modelos pre-entrenados:
            * Selección modelo: BERT, GPT, T5, RoBERTa y variantes especializadas.
            * Tokenization and input preparation.
            * Inferencia para tareas comunes: clasificación, generación, respuesta a preguntas.
        * **Fine-tuning:** Adaptar modelos pre-entrenados a tareas específicas.
            * Transfer learning in NLP.
            * Estrategias de capacitación e hiperparametros.
            * Financiamiento eficiente: LoRA y métodos eficientes del parámetro.
    * **Transformers Beyond NLP:**
        * **Pronóstico de series temporales:** Atención temporal para predicciones.
        * Transformadores de visión (ViT): aplicar transformadores a imágenes.
        * Transformadores multimodales: combinando texto, imágenes y otras modalidades.

-..

### **Proyecto 8: Evolución de las arquitecturas NLP - Desde el ML tradicional a los transformadores**
* Aplicado después de la Unidad 8 - Datos y Transformadores secuenciales*
* **Objetivo:** Comparación amplia de las arquitecturas de NLP mediante la evolución de los enfoques de clasificación de textos.
* ** Componentes clave:**
    * **Problema persistente** Detección de noticias falsa en todas las fases para la comparación directa.
    * **Phase 1 - Tradicional ML:** Características TF-IDF con SVM, Naive Bayes y métodos de ensemble.
    * **Fase 2 - Redes neuronales:** Perceptores multicapa con incrustaciones de palabras.
    * **Página 3 - CNNs:** Redes revolucionarias para clasificación de textos con diferentes tamaños de filtros.
    * **Phase 4 - RNNs:** LSTM y redes bidireccionales LSTM para el procesamiento secuencial.
    * **Phase 5 - Transformers:** Modelos BERT finos para el rendimiento de última generación.
    * **Análisis avanzado** Comparación de rendimiento, análisis de costos computacionales y estudio de interpretabilidad.
    * ** Aplicación práctica:** API de detección de noticias falsas con múltiples opciones de modelo.
* ** Resultados de aprendizaje:** Comprender la evolución de NLP y cuándo utilizar cada arquitectura.
* **Nota:** Utiliza bibliotecas de alto nivel (scikit-learn, PyTorch, Hugging Face) para centrarse en la comparación de arquitectura.

-..

### **Unit 9: Generative AI and Large Language Models (LLMs)**
* **Objetivo:** Entender y utilizar modelos capaces de generar contenido nuevo y coherente.
* **Contenidos:**
    * ** Modelos Generativos Paisaje:**
        * ¿Qué es IA Generativa? Panorama general de diferentes enfoques.
        * ** Redes adversarias de carácter consultivo (GAN):**
            * Generador vs Discriminator: concepto de entrenamiento adversario.
            * Aplicaciones: generación de imágenes, aumento de datos, transferencia de estilo.
            * Problemas de entrenamiento: colapso del modo, problemas de convergencia.
            * Importancia histórica y casos de uso actual.
        * ** Modelos autoregresivos:**
            * Generación secuencial: predecir el siguiente token basado en anteriores.
            * Ejemplos de arquitectura: familia GPT, modelos de lenguaje tradicionales.
            * Aplicaciones: generación de texto, terminación de código, pronóstico de series temporales.
            * Fuerza: generación coherente de larga duración, flexibilidad.
        * ** Modelos de difusión:**
            * Proceso de avance: añadiendo gradualmente el ruido a los datos.
            * Proceso inverso: aprender a denoizar y generar muestras.
            * Aplicaciones modernas: DALL-E 2, Midjourney, Stable Diffusion.
            * Ventajas sobre los GAN: estabilidad de entrenamiento, calidad de muestra, cobertura de modo.
        * ** Comparación de enfoques generativos:** Cuando usar cada paradigma.
    * **Modelos de idioma alto (LLM):**
        * Fundación de arquitectura: Modelos autoregresivos basados en transformadores (GPT).
        * Escala y emergencia: cómo el tamaño permite nuevas capacidades.
        * paradigma de pre-entrenamiento y ajuste.
    * **Prompt Engineering:** El arte de diseñar insumos eficaces para LLMs.
    * ** Desarrollo de la aplicación de la LLM**
        * ** Enfoque directo**: API nativas (OpenAI, Antropopic) para el máximo control y transparencia.
        * ** Marcos de orquestación LLM:**
            * **LangChain**: Marco de uso general para la construcción de aplicaciones de LLM.
                * Cadenas: Composición de llamadas LLM y lógica.
                * Agentes: Selección dinámica de herramientas y razonamiento.
                * Memoria: Historia de la conversación y gestión del contexto.
                * Cargadores de documentos y separadores de texto.
            * **LlamaIndex**: Marco especializado para RAG y recuperación de conocimientos.
                * Motores de construcción de índices y consultas.
                * Estrategias avanzadas de recuperación.
                * Integración con tiendas vectoriales.
            * Comparación**: Cuando utilizar cada marco y cómo se complementan.
        * Gestión de memoria y contexto en aplicaciones de conversación.
    * ** Bases de datos de vehículos e incrustaciones:**
        * Conceptos: búsqueda de similitudes, incrustaciones de alta dimensión.
        * **ChromaDB**: Base de datos de vectores práctica y fácil de usar.
        * ** Aplicación práctica**: Implementación de búsqueda semántica y RAG con documentos reales.
    * **RAG (generación aumentada de recuperación):** Aumentar el conocimiento de LLM con documentos externos para reducir alucinaciones y utilizar datos privados.
    * **Agents and Model Context Protocol:**
        * Crear sistemas autónomos que utilicen LLMs para razonar y ejecutar acciones.
        * **Model Context Protocol (MCP):** Estándar para conectar LLMs con herramientas y datos externos.
        * Implementación de agentes con capacidades de interacción avanzadas.

-..

### **Proyecto 9: Procesamiento de documentos inteligentes con RAG**
* Aplicado después de la unidad 9 - Generative AI*
* **Objetivo:** Sistema RAG completo para análisis de documentos y respuesta de preguntas.
* ** Componentes clave:**
    * ** Ingestión de documentos:** PDF, Word, web scraping para diversas fuentes de documentos.
    * **Procesamiento de texto:** Estrategias de caza, generación de incrustación con diferentes modelos.
    * ** Almacenamiento de vehículos:** Implementación de cromaDB con filtrado de metadatos.
    * **Grupo de control:** Procesamiento de consultas, recuperación y generación con LLMs.
    * **Evaluación** Precisión de recuperación, calidad de respuesta y detección de alucinaciones.
    * ** Aplicación práctica:** Base de conocimientos corporativos o sistema legal de análisis de documentos.
* ** Características avanzadas:** RAG multimodal con imágenes, memoria de conversación e integración de agentes.


-..

### **Unit 10: MLOps and Production Machine Learning**
* **Objetivo:** Domine las prácticas y herramientas para el despliegue, monitoreo y mantenimiento de modelos ML en entornos de producción.
* **Motivación:** Bridging the gap between model development and real-world deployment - the most critical skill for ML practitioners.
* **Contenidos:**
    * **El paisaje de los MLOps:**
        * ¿Qué es MLOps? Principios de DevOps aplicados al ML.
        * The ML production lifecycle: development, deployment, monitoring, retraining.
        * Desafíos comunes: deriva modelo, deriva de datos, reproducibilidad, escalado.
        * Niveles de madurez MLOps: de manual a totalmente automatizado.
    * **Experiment Tracking and Model Management:**
        * ** Flujo de sol** Seguimiento de experimentos, parámetros, métricas y artefactos.
        * **Pesas " . Seguimiento y visualización de experimentos avanzados.
        * Registro modelo: Versioning models and managing their lifecycle.
        * Comparación de experimentos: Encontrar el mejor modelo sistemáticamente.
        * Reproducibilidad: Asegurar experimentos se puede replicar.
    * ** Estrategias de despliegue modelo:**
        * **Containerization with Docker:**
            * Creación de imágenes Docker para modelos ML.
            * Dockerfile mejores prácticas para aplicaciones ML.
            * Gestión de las dependencias y la coherencia del medio ambiente.
        * **API Development:**
            * **FastAPI:** Creación de puntos finales RESTful para la inferencia modelo.
            * Solicitar validación, manejo de errores y documentación.
            * Sincronía vs inferencia asincrónica.
            * Puntos finales de predicción de lotes.
        * ** Patrones de despliegue**
            * Despliegue verde azul para cero tiempo de inactividad.
            * Las versiones canarias para el despliegue gradual.
            * Pruebas A/B para comparación de modelos.
            * Modo de sombra para pruebas seguras.
    * ** Despliegue en voz alta**
        * Vista general de las plataformas ML de la nube:
            * **AWS SageMaker:** Plataforma ML final a extremo.
            * **Aprendizaje de Máquina Azul:** La suite ML de Microsoft.
            * **Google Cloud Vertex AI:** Plataforma ML unificada.
        * Implementación sin servidor: AWS Lambda, Funciones Azure, Funciones de Google Cloud.
        * Puntos finales gestionados y escalado automático.
        * Estrategias de optimización de costos.
    * **Model Monitoring and Observability:**
        * ** Supervisión de la ejecución** Seguimiento de latencia, rendimiento, tasas de error.
        * **Detección de deriva modelo:**
            * Desvío de datos: Cambios en las distribuciones de insumos.
            * Concepto deriva: Cambios en la relación entre entradas y salidas.
            * Herramientas: Evidently AI, NannyML, Alibi Detect.
        * **Alerting systems:** Configuración de alertas para el rendimiento degradado.
        * **Logging best practices:** Registro estructurado para sistemas ML.
        * **Dibujos de medición:** Grafana, Prometheus para el monitoreo de ML.
    * **CI/CD para el aprendizaje automático:**
        * Integración continua: Pruebas automatizadas para código y datos ML.
        * Despliegue continuo: tuberías de despliegue de modelos automatizadas.
        * **GitHub Actions:** Configuración de tuberías ML.
        * Testing ML systems:
            * Pruebas de unidad para el procesamiento de datos e ingeniería de características.
            * Pruebas de integración para servir modelo.
            * Pruebas de humo para modelos desplegados.
        * ** validación de datos:** Grandes expectativas para cheques de calidad de datos.
    * **Model Versioning and Governance:**
        * **Control de la versión de datos (DVC):** Configuración de datos y modelos.
        * Lineaje modelo: Rastreo de orígenes y transformaciones modelo.
        * Documentación modelo y metadatos.
        * Senderos de cumplimiento y auditoría.
    * **Scaling ML Systems:**
        * Escalada horizontal vs vertical.
        * Equilibrio de carga para inferencia modelo.
        * Estrategias de captura para predicciones comunes.
        * Batch vs en tiempo real.
        * *Kubernetes for ML* Fundamentos de orquestación de contenedores.
    * ** Temas avanzados:**
        * **Edge deployment:** TensorFlow Lite, ONNX para móvil e IoT.
        * ** Optimización modelo:** Cuantización, poda, destilación.
        * **Multi-model servir:** Hosting múltiples modelos eficientemente.
        * **Características:** Gestión centralizada de funciones (Feast, Tecton).


-..

## Temas transversales (a tener en cuenta pero no como unidad separada)

### Ética e interpretación en AI
    * **Bias y equidad:**
        * Fuentes de sesgo: Sesgo de datos, sesgo algorítmico, sesgo de interacción.
        * **Tipos de equidad:**
            * La equidad individual frente a la equidad de grupo.
            * Paridad demográfica, probabilidades igualadas, calibración.
        * **Detección de Bias:** Pruebas estadísticas y métricas de imparcialidad.
        * **Bias mitigation:**
            * Pre-procesamiento: Repeso, estrategias de muestreo.
            * En proceso: algoritmos de aprendizaje justo.
            * Post-procesamiento: Ajuste del umbral.
        * **Herramientas: AI Fairness 360, Fairlearn.
    * ** Interpretabilidad y Explicabilidad Modelo:**
        * **Por qué la interpretación importa:** Confianza, depuración, cumplimiento, descubrimiento científico.
        * **Global vs local interpretationability:**
            * Global: Comprender el comportamiento general del modelo.
            * Local: Explicando predicciones individuales.
        * ** Modelos intrínsecamente interpretables:**
            * Árboles de decisiones, modelos lineales, sistemas basados en reglas.
            * Cambios: Interpretabilidad vs rendimiento.
        * ** Métodos de explicación post-hoc:**
            * **SHAP (SHapley Additive exPlanations):** Marco unificado para explicaciones modelo.
            * **LIME (Explicaciones modelo-agnósticas interpretables locales):** Modelos locales de sustitución.
            * ** Visualización de la atención:** Comprender en qué se enfocan los transformadores.
            * ** Mapas de emergencia:** Visualización de importantes regiones en imágenes.
        * ** Herramientas prácticas:** SHAP, LIME, Captum (para PyTorch), InterpretML.
   

-..

## **Temáticas avanzadas opcionales** (Permiso de tiempo o autoestudio)

### **Forecasting Series Time**
* ARIMA, SARIMA y métodos clásicos.
* Profeta para la serie de tiempo de negocios.
* Pronóstico basado en transformadores (Transformador de Fusión Temporal).
* Aplicaciones: Predicción de la demanda, predicciones financieras, detección de anomalías en series temporales.

### **Multimodal AI**
* CLIP: Conexión de la visión y el lenguaje.
* Modelos de lenguaje de visión para capturar imágenes y VQA.
* Tratamiento de audio y reconocimiento de discursos.
* Embeddings multimodales y aplicaciones.

### ** Temas avanzados de aprendizaje profundo**
* Graph Neural Networks (GNNs) for graph-structured data.
* Aprendizaje de meta-aprendizaje y poca monta.
* Técnicas de aprendizaje autosupervisadas.
* Neural Architecture Search (NAS).

### ** Técnicas avanzadas de LLM**
* **Fine-tuning at scale:** LoRA, QLoRA, PEFT en profundidad.
* **Prompt engineering:** Técnicas avanzadas, cadena de pensamiento, aprendizaje corto.
* ** Evaluación de la LLM** Parámetros, evaluación humana, métrica automatizada.
* **IA constitucional y RLHF** Alinear modelos con valores humanos.

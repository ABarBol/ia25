# La revolución de la IA Generativa

Mientras que la IA tradicional (o la IA predictiva) se centra en analizar los datos existentes para hacer predicciones o clasificaciones, **La IA Generante** es sobre crear contenido nuevo y original**. Esto marca un cambio fundamental de entender el mundo a crear dentro de él.

Los modelos generadores aprenden los patrones subyacentes y la estructura de un conjunto de datos y luego utilizan ese conocimiento para producir artefactos novedosos que son similares a, pero distintos de, los datos de entrenamiento.

## Arquitecturas generadoras clave

Varias arquitecturas poderosas han impulsado el boom generativo de la IA.

### 1. Generative Adversarial Networks (GANs)

- **Concepto**: Los GAN consisten en dos redes neuronales, un **Generador** y un **Discriminador**, encerrados en un juego competitivo.
    - El **Generador** crea datos falsos (por ejemplo, imágenes, texto).
    - El **Discriminador** trata de distinguir entre los datos reales y los datos falsos creados por el generador.
- **El juego**: El objetivo del generador es engañar al discriminador, mientras que el objetivo del discriminador es mejorar la captura de falsificaciones. A través de este proceso contradictorio, el generador se vuelve progresivamente mejor creando contenido realista.
- **Impact**: GANs revolucionó la generación de imágenes, permitiendo la creación de caras fotorrealistas, arte y otros contenidos visuales.

### 2. Autoencoders Variational (VAEs)

- **Concepto**: Las VAEs son una extensión de autoencoders. Aprenden a comprimir los datos en un "espacio latente" estructurado y de menor dimensión y luego reconstruirlos.
- **Proceso Generativo**: Al mostrar puntos de este espacio latente aprendido, un VAE puede generar nuevos datos. La estructura del espacio latente permite una interpolación fluida entre diferentes puntos de datos (por ejemplo, morphing one face into another).
- **Use Casos**: Los VAEs se utilizan para la generación de imágenes, el aumento de datos y la detección de anomalías.

### 3. Modelos de difusión

- **Concepto**: Los modelos de difusión funcionan agregando sistemáticamente ruido a una imagen hasta que se convierte en estática pura, y luego aprendiendo a revertir el proceso.
- **La inversión**: El modelo aprende a eliminar gradualmente el ruido, paso a paso, para recuperar una imagen limpia de una entrada de ruido al azar. Al controlar el proceso con impulsos de texto u otras entradas, puede generar imágenes altamente específicas y detalladas.
- **Impact**: Los modelos de difusión son el vanguardia de la generación de imágenes, modelos de potencia como **DALL-E 2/3, Midjourney y Stable Diffusion**. Producen imágenes de impresionante calidad y coherencia.

### 4. Transformers and Large Language Models (LLMs)

- **Concepto**: La arquitectura **Transformer**, introducida en el documento de 2017 "Attention is all you Need", revolucionó cómo procesamos datos secuenciales como texto. Su innovación clave es el mecanismo de autoatención **, que permite al modelo pesar la importancia de las diferentes palabras en una secuencia al procesarlo.
- **Modelos de idioma alto (LLM)**: Al ampliar los modelos Transformer a miles de millones de parámetros y entrenarlos en vastas cantidades de datos de texto de Internet, los investigadores crearon LLMs como **GPT (Generative Pre-trained Transformer)**.
- **Cómo funcionan**: Los LLM son fundamentalmente **Los predictores de los próximos documentos**. Dada una secuencia de texto, predicen la siguiente palabra o carácter más probable. Al hacer esto repetidamente, pueden generar pasajes largos y coherentes de texto.
- **Impact**: LLMs ha desbloqueado una gran variedad de aplicaciones, desde chatbots y creación de contenidos a generación de códigos y descubrimiento científico. Representan un paso importante hacia una inteligencia artificial más general.

## Patrones prácticos de aplicación de LLM

- Instrucciones del sistema y de protagonismo: el comportamiento del control con papeles claros, restricciones y ejemplos (fro-shot).
- Función de llamada / uso de herramientas: limitar salidas a esquemas JSON y llamar herramientas externas de forma segura.
- Generación aumentada de recuperación (RAG): enriquecer los impulsos con contexto recuperado de datos privados (ver más abajo).
- Guardias y validación: validación de esquemas, filtros de contenido y controles de seguridad.

### Generación aumentada en recuperación (RAG)

RAG reduce alucinaciones y permite el uso del conocimiento privado combinando la recuperación con la generación.

Gasoducto básico:
- Ingestión: documentos en pedazos; limpiar y normalizar texto.
- Embeddings: pedazos de mapa a vectores con un modelo de embedding.
- Vector store: vectores índice en una base de datos de búsqueda de similitudes.
- Retrieval: para una consulta, recupere los trozos relevantes de alta-k; opcionalmente re-rank.
- Síntesis: construir un impulso con el contexto recuperado; generar la respuesta.

Notas y obstáculos:
- Cuestiones de la estrategia de Chunking (partición semántica, superposición para preservar el contexto).
- Evaluar la recuperación (precision@k, recall@k) y el QA final-a-final (respuesta de fidelidad, fundamento).
- Añadir citas: adjuntar las fuentes a respuestas para la verificabilidad.
- Caching y deduplicación reducen el costo y el ruido.

Minimal “contrato” para una tarea RAG QA:
- Entrada: consulta de usuario; filtros/metadatos opcionales.
- Producto: texto de respuesta; fuentes [{doc id, span}]], confianza.
- Errores: cuerpo vacío, no partes relevantes, límites de tarifas.

Las bases de datos vectoriales suelen proporcionar: índices HNSW/IVF, filtros de metadatos, búsqueda híbrida (sparse+dense), y persistencia.


## La subida de la AI Agentic

La última evolución en AI es el desarrollo de **Agentes**. Un agente de IA es un sistema que va más allá de responder pasivamente a los impulsos y puede ** razonar, planificar y ejecutar tareas de manera autónoma** para lograr un objetivo.

### Componentes básicos de un agente de inteligencia artificial

1.  **LLM como el "Brain"**: Un poderoso modelo de lenguaje grande sirve como el motor de razonamiento central.
2.  **Planning**: El agente descompone un objetivo de alto nivel en una secuencia de pasos más pequeños y factibles.
3.  **Tool Use**: El agente tiene acceso a un conjunto de herramientas (por ejemplo, una búsqueda web, un intérprete de código, otras API) que puede decidir utilizar para recopilar información o realizar acciones.
4.  *Memoria* El agente mantiene un recuerdo de sus acciones, observaciones y auto-reflexiones para aprender de su experiencia y adaptar su plan.

### Cómo funcionan los agentes: el marco de acción

Un marco popular para los agentes de construcción es **React (Reason + Act)**. En este ciclo, el agente:
1.  ** Razones** sobre el estado actual y el objetivo general.
2.  Decide tomar **Acción** (por ejemplo, utilizar una herramienta, responder al usuario).
3.  Hace un **Observación** basado en el resultado de su acción.
4.  Repita el ciclo, utilizando la nueva observación para informar su próximo paso de razonamiento.

Este circuito iterativo permite que el agente adapte dinámicamente su plan basado en nueva información, lo que hace mucho más poderoso y flexible que un simple sistema de respuesta rápida.

Otros patrones útiles:
- Plan-y-Ejecutar: división de la planificación de alto nivel de la ejecución de bajo nivel para tareas largas.
- Autoinvocación de estilo Toolformer: los modelos deciden cuándo llamar herramientas.
- Razonamiento deliberado: asigne una fase separada de “pensamiento” antes de actuar (con o sin visibilidad).

### Model Context Protocol (MCP)

MCP es un protocolo abierto que estandariza cómo las LLM se conectan a herramientas, fuentes de datos y acciones.

Ideas clave:
- Los servidores exponen las capacidades (herramientas, impulsos, recursos) mediante un contrato escrito.
- Los clientes (tiempos de ejecución/edidores) descubren e invocan esas capacidades.
- Ventajas: portabilidad de herramientas a través de modelos, interacciones auditables, código de pegamento reducido.

Componentes típicos:
- Registro de responsabilidad: lista de herramientas disponibles y esquemas.
- Canal de invocación: solicitud/respuesta para llamadas de herramientas con OI estructurado.
- Acceso a recursos: buscar documentos, archivos o bases de datos a través de una interfaz unificada.

Consejos de diseño:
- Mantén los esquemas IO pequeños y validados; favore las operaciones idempotent.
- Proporcionar herramientas apátridas cuando sea posible; externalizar el estado a una tienda.
- Lograr todas las llamadas para las auditorías de observabilidad y seguridad.

Ilustración (proporcionada):

![Resumen del PCM](./img/mcp.gif)

### AI Gateway

Una puerta de entrada AI se encuentra entre su aplicación y uno o más proveedores de modelos.

Características principales:
- Routing and fallbacks across providers/models; A/B testing and canaries.
- Observabilidad: rastreo, contabilidad de token/cost, latencia, errores.
- Seguridad: filtros de contenido, reacción rápida, validación de salida.
- Controles de políticas: límites de tarifas, cuotas, manejo de PII.
- Prompt and template management; caching and response deduplication.

Cuándo utilizar:
- Múltiples proveedores o modelos; optimización de costos/rendimiento; gobernanza centralizada.


### El futuro es un agente

Agentic AI representa un cambio de paradigma donde interactuamos con AI no sólo como una herramienta, sino como un ** socio colaborativo**. Estos agentes pueden manejar tareas complejas y de varios pasos, desde la planificación de un viaje y reservar vuelos hasta la realización de software de investigación científica y escritura. El desarrollo de agentes más capaces y autónomos es un área clave de la investigación actual de IA y está preparado para redefinir cómo trabajamos e interactuamos con la tecnología.


# ¿Qué es Inteligencia Artificial? Introducción histórica y conceptual

## Definición de inteligencia: La búsqueda de emular la mente humana

Inteligencia Artificial (AI) es un campo amplio y ambicioso de la informática dedicado a crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estas tareas incluyen razonamiento, aprendizaje, solución de problemas, percepción, comprensión del lenguaje e incluso creatividad.

En su núcleo, AI no se trata sólo de procesar datos sino de **anunciar cognición**. El objetivo final es construir máquinas que puedan pensar, aprender y adaptarse de formas indistinguibles de capacidades humanas o incluso superiores a ellas.

### The Turing Test: A Benchmark for Intelligence

En 1950, Alan Turing propuso una prueba simple pero poderosa para medir la capacidad de una máquina para exhibir comportamiento inteligente equivalente a, o indistinguible de, el de un humano. En el **Turing Test**, un evaluador humano participa en una conversación de lenguaje natural con un humano y una máquina. Si el evaluador no puede confiar en decirle a la máquina del humano, se dice que la máquina ha pasado la prueba.


## Una breve historia de AI

El sueño de crear seres artificiales es antiguo, pero el viaje científico de AI comenzó a mediados del siglo XX.

- **1956: The Dartmouth Workshop - The Birth of a Field**: The term "Artificial Intelligence" was coined by John McCarthy at a summer workshop at Dartmouth College. Este evento reunió a los padres fundadores de AI y plantó la visión para el campo. El optimismo inicial fue inmenso, y los pioneros predijeron que las máquinas con inteligencia a nivel humano estaban a pocas décadas de distancia.

- **Los "AI Winters": Ciclos de Hipo y Desilusión**: La historia de AI ha estado marcada por periodos de intensa financiación y emoción ("verano AI") seguidos de "inviernos AI", donde el progreso se estancó y la financiación secó. Estos ciclos fueron a menudo causados por:
    - *Promesas obstruidas* Los primeros investigadores subestimaron la profunda dificultad de las tareas como la visión informática y la comprensión del lenguaje natural.
    - ** Límites Computacionales**: El hardware del tiempo era insuficiente para manejar la complejidad de los modelos propuestos.
    - **La Explosión Combinatorial**: Muchos enfoques tempranos de IA se basaron en explorar vastos espacios de búsqueda de posibilidades, que rápidamente se convirtieron en computacionalmente intráctil.

- **2012-Present: La Revolución del Aprendizaje Profundo**: El actual "Verano AI" fue encendido por la convergencia de tres factores clave:
    1.  **Big Data**: La disponibilidad de conjuntos de datos masivos para formar modelos complejos.
    2.  **Powerful Hardware**: El aumento de las GPU (unidades de procesamiento de gráficos) proporcionó el poder de cálculo paralelo necesario para el aprendizaje profundo.
    3.  ** Avances algorítmicos**: Las innovaciones como el algoritmo de retropropagación y las nuevas arquitecturas de redes neuronales (por ejemplo, AlexNet en 2012) desbloquearon un rendimiento sin precedentes.

Esta revolución cambió el paradigma dominante de los sistemas basados en reglas a ** machine learning**, donde los sistemas aprenden directamente de los datos.

## Machine Learning vs. Artificial Intelligence

Mientras que a menudo se utiliza intercambiablemente, **Machine Learning (ML)** es un subcampo de AI que es más fácil de definir. Se centra en sistemas de construcción que pueden **aprender de datos**, identificar patrones y tomar decisiones con mínima intervención humana.

<img src="img/ai_hierarchy.png" width="200" style="margin: 20px;">

En lugar de estar explícitamente programado con reglas para resolver un problema, un modelo de aprendizaje automático aprende su propio algoritmo analizando y encontrando patrones en los datos. Cuanto más datos esté expuesto, mejor será el modelo.

Un par de definiciones clásicas son:

> [El aprendizaje mecánico es el] campo de estudio que da a los ordenadores la capacidad de aprender sin ser explícitamente programado.
>

> — Arthur Samuel, 1959

> Se dice que un programa informático aprende de la experiencia E con respecto a alguna clase de tareas T y medida de rendimiento P si su rendimiento en las tareas en T, medido por P, mejora con la experiencia E.
>

> Tom Mitchell, 1997

### Aprendizaje profundo

**Aprendizaje profundo** es un subcampo especializado de aprendizaje automático que utiliza redes neuronales artísticas** con muchas capas (de ahí "deep"). Al aprovechar arquitecturas profundas, estos modelos pueden aprender patrones complejos y jerárquicos de vastas cantidades de datos. Esto ha llevado a avances en campos como la visión informática y el procesamiento de lenguaje natural.

## El espectro de AI: De la inteligencia estrecha a la inteligencia general

Los sistemas de IA pueden clasificarse en función de sus capacidades y nivel de "consciencia"

### Weak AI (Narrow AI)

**Weak AI**, también conocido como **Artificial Narrow Intelligence (ANI)**, se refiere a sistemas AI diseñados y entrenados para realizar una tarea ** específica y bien definida**. Esta es la forma de AI que nos rodea hoy.

- *Características*
    - **Task-Specific**: Excels en un trabajo (por ejemplo, jugar ajedrez, reconocer rostros, filtrar spam).
    - ** Ninguna conciencia o conciencia de sí mismo**: Funciona dentro de un rango predeterminado y no posee una comprensión o conciencia genuinas.
    - **Data-Driven**: Su rendimiento está directamente ligado a la calidad y cantidad de los datos en los que fue entrenado.

- **Ejemplos**:
    - **Siri, Alexa y Google Assistant**: Asistentes activados por voz que entienden y responden a un conjunto limitado de comandos.
    - **Recomendation Engines**: Algorithms on Netflix o Amazon que sugieren contenido basado en su historia visual.
    - ** Autoconducir coches**: Sistemas altamente complejos, pero aún estrecho AI se centró en la tarea de conducir.

### Strong AI (General AI o AGI)

**Strong AI**, o **Artificial General Intelligence (AGI)**, es la inteligencia hipotética de una máquina que tiene la capacidad de entender, aprender y aplicar su inteligencia para resolver **cualquier tarea intelectual que un ser humano pueda**.

- *Características*
    - *Cognición humana* Posee la capacidad de razonar, planificar, aprender de la experiencia, pensar abstractamente y entender ideas complejas.
    - **Conciencia y conciencia de sí mismo**: Un verdadero AGI probablemente tendría alguna forma de conciencia y experiencia subjetiva (aunque este es un tema de intenso debate filosófico).
    - **Adaptability**: Puede transferir conocimientos de un dominio a otro y aprender nuevas tareas sin ser explícitamente reprogramado.

**AGI sigue siendo el santo gral de investigación de IA y aún no existe.**

### Superinteligencia Artificial (ASI)

**La Superinteligencia Artística (ASI)** es una forma hipotética de AI que sobrepasa la inteligencia humana en prácticamente todos los dominios, incluyendo creatividad científica, sabiduría general y habilidades sociales. El desarrollo de ASI plantea profundas cuestiones éticas y existenciales para el futuro de la humanidad.


# Herramientas esenciales para la programación de AI

## Idiomas de programación

### Python

Python es el ** lenguaje de programación predominante para * machine learning***. Será el idioma que usamos a lo largo de este curso. Python es un lenguaje de programación interpretado, de alto nivel y para fines generales. Es un lenguaje de programación muy versátil y fácil de aprender, y tiene una colección **vasta de bibliotecas y marcos para AI**, * machine learning* y *deep learning*.

Cursos de programación de pitón:
- https://edube.org/study/pe1
- https://edube.org/study/pe2


### Otros idiomas relevantes de programación en AI

- R: Principalmente utilizado en estadísticas y análisis de datos.
- C/C++: Muchas de las bibliotecas a las que se accede desde Python se implementan en C/C++ ya que son idiomas de bajo nivel y mucho más rápido que Python. C++ sigue siendo la referencia cuando el rendimiento es crítico.
- Julia: Lenguaje de programación de alto rendimiento diseñado para la informática científica y técnica.
- Mojo: Objetivos para lograr la usabilidad de Python con rendimiento similar a C.


## Git y Git Forges

Git es un sistema de control de versiones distribuido que nos permite mantener un registro de cambios en el código fuente de un proyecto.
Las forjas Git son servicios que nos permiten acoger nuestros repositorios Git en la nube. Algunas de las forjas más populares son GitHub, GitLab y Bitbucket. A lo largo de este curso, utilizaremos ampliamente Git y GitHub.

- [Git y GitHub Resources](https://github.com/avidaldo/recursos-git)
- [Git and GitHub for AI Programming](https://www.youtube.com/watch?v=T1tYBbiWTbc)


## IDEs (Integrated Development Environment)

Para trabajar con Python para * machine learning* y *deep learning*, algunos de los IDE más populares son:

### [Visual Studio Code (VS Code)](https://code.visualstudio.com/)

Visual Studio Code es un editor de código fuente desarrollado por Microsoft. Es multiplataforma y admite múltiples lenguajes de programación. Es uno de los editores más populares entre desarrolladores de software.

- [CODE DE ESTUDIO VISUAL: Tutorial del principiante](https://www.youtube.com/watch?v=CxF3ykWP1H4)

### [Cursor](https://www.cursor.com/)

Cursor es un [hork](https://en.wikipedia.org/wiki/Fork_(software_development)) de código VS centrado en el uso de LLMs para la programación.

### [PyCharm](https://www.jetbrains.com/pycharm/)

PyCharm es un IDE desarrollado por JetBrains. Es uno de los IDE más populares para Python y es utilizado por muchos desarrolladores de software.

### [DataSpell](https://www.jetbrains.com/dataspell/)

DataSpell es otro JetBrains IDE similar al PyCharm pero especializado en análisis de datos.

## Jupyter Notebooks

Jupyter es una aplicación que nos permite crear *notebooks* (documentos que combinan código, texto rico, ecuaciones, visualizaciones, etc.) en el navegador. Estos *notebooks* son archivos con la extensión .ipynb (al utilizar el lenguaje Python) que se pueden exportar en diferentes formatos estáticos, como HTML o PDF. Los archivos `.ipynb` son internamente JSON.

## Entornos virtuales

**Entornos virtuales** nos permiten aislar las **dependencias de un proyecto** de las del sistema operativo. Esto nos permite tener diferentes versiones de las mismas **bibliografías** en diferentes proyectos sin conflictos entre ellos.

 - [Conda Tutorial: Medio Ambiente y Gestión de Paquetes](conda_tutorial.md)
 - [tutorial uv: un instalador de paquetes rápido de pitón y resolver](uv_tutorial.md)

 - [Introducción a entornos de pip, conda y virtual](https://www.youtube.com/watch?v=7Rd-Gj8o-6Q)

Utilizaremos principalmente el `uv` más reciente; una herramienta moderna para gestionar entornos virtuales y dependencias en proyectos de Python.

 - [UV para Python... (casi) Todas las baterías incluidas](https://www.youtube.com/watch?v=qh98qOND6MI)

### [Dependencia y Gestión Virtual del Medio Ambiente en Python](py_dependencies.md)


## [Google Colab](https://colab.research.google.com/)

Google Colab es un servicio gratuito de Google que nos permite ejecutar Jupyter *notebooks* en la nube. Colab nos permite ejecutar código en Python, R y otros idiomas de programación, y proporciona acceso a recursos de hardware como GPUs y TPU.


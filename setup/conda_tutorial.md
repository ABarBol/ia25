# Conda Tutorial: Medio Ambiente y Gestión de Paquetes

Conda es un sistema de gestión de paquetes y entornos de código abierto que funciona en Windows, macOS y Linux. Es ampliamente utilizado en las comunidades de ciencia de datos y aprendizaje automático.

## 1. Instalación

Recomendamos instalar **Miniconda**, un instalador mínimo para conda. Es más pequeño y más rápido instalar que la distribución completa de Anaconda.

1.  **Descargue el instalador**: Vaya al [Miniconda página de descarga](https://docs.conda.io/en/latest/miniconda.html) y descargue el instalador adecuado para su sistema operativo.

2.  **Arranque el instalador**: Siga las instrucciones para su sistema operativo. Se recomienda permitir que el instalador inicialice conda, lo que lo hará disponible en su terminal.

3.  **Verificar la instalación**: Abrir una nueva terminal y ejecutar:
    ```bash
    conda --version
    ```
    Deberías ver la versión conda instalada.

## 2. Managing Environments

Un entorno conda es un directorio que contiene una colección específica de paquetes de conda que ha instalado.

### Crear un entorno

Para crear un nuevo entorno, utilice `conda create`. Es buena práctica especificar la versión Python que desea utilizar.

```bash
# Creates an environment named 'myenv' with Python 3.12
conda create --name myenv python=3.12
```

También puede instalar paquetes al mismo tiempo:

```bash
conda create --name myenv python=3.12 numpy pandas jupyter
```

### Activar y desactivar un entorno

Antes de que puedas usar un ambiente, necesitas activarlo.

```bash
# Activate the environment
conda activate myenv
```

Su impulso terminal debe cambiar para mostrar el nombre del entorno activo.

Desactivar el entorno actual y volver al entorno base:

```bash
# Deactivate the environment
conda deactivate
```

### List Environments

Para ver una lista de todos sus entornos:

```bash
conda env list
```

El entorno activo estará marcado con un asterisco (*).

### Remove an Environment

Para eliminar un entorno y todos los paquetes instalados en él:

```bash
# Make sure the environment is not active
conda deactivate

# Remove the environment
conda env remove --name myenv
```

## 3. Gestión de paquetes

Con su entorno activado, puede instalar, actualizar y eliminar paquetes.

### Instalar paquetes

Utilice `conda install` para instalar paquetes desde el canal Anaconda predeterminado.

```bash
# Activate your environment first
conda activate myenv

# Install packages
conda install numpy pandas scikit-learn
```

También puede especificar versiones de paquetes:

```bash
conda install numpy=1.26.0
```

### Instalar desde otros canales

A veces los paquetes no están disponibles en el canal predeterminado. Un canal popular es `conda-forge`.

```bash
conda install -c conda-forge some-package
```

### Lista de paquetes instalados

Para ver todos los paquetes instalados en el entorno actual:

```bash
conda list
```

### Paquetes de actualización

Para actualizar un paquete específico:

```bash
conda update numpy
```

Para actualizar todos los paquetes en el medio ambiente:

```bash
conda update --all
```

### Eliminar paquetes

Para desinstalar un paquete:

```bash
conda remove scikit-learn
```

## 4. Compartir entornos

Puede compartir su entorno con otros exportando su especificación a un archivo YAML.

### Exportar un medio ambiente

1.  Activar el entorno que desea exportar.
    ```bash
    conda activate myenv
    ```
2.  Exportar el entorno a un archivo (comúnmente llamado `environment.yml`).
    ```bash
    conda env export > environment.yml
    ```

Este archivo puede ser compartido, y otros pueden replicar su entorno.

### Crear un entorno desde un archivo

Para crear un entorno desde un archivo `environment.yml`:

```bash
conda env create -f environment.yml
```

Esto creará un nuevo entorno con el mismo nombre y paquetes especificados en el archivo.

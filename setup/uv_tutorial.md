# tutorial uv: un instalador de paquetes rápido de pitón y resolver

`uv` es un instalador y resolución de paquetes Python extremadamente rápido, escrito en Rust. Está diseñado como un reemplazo para `pip` y `venv`.

## 1. Instalación

Puede instalar `uv` utilizando `curl` en macOS y Linux, o `pip` si ya tiene instalado Python.

**Using curl (macOS, Linux):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Usando PowerShell (Windows):**
```powershell
irm https://astral.sh/uv/install.ps1 | iwr
```

*Using pip*
```bash
pip install uv
```

Verificar la instalación:
```bash
uv --version
```

## 2. Gestión de los entornos virtuales

`uv` integra la gestión del medio ambiente virtual, similar a `venv`.

### Crear un entorno virtual

Para crear un nuevo entorno virtual, utilice `uv venv`. Por defecto, crea un directorio `.venv` en su ubicación actual.

```bash
# Create a virtual environment in the .venv directory
uv venv
```

También puede especificar una versión Python si tiene varias versiones instaladas:

```bash
# Create a virtual environment with Python 3.12
uv venv -p 3.12
```

### Activar y desactivar el medio ambiente

Para activar el entorno virtual, necesita ejecutar el script de activación.

**En macOS y Linux:**
```bash
source .venv/bin/activate
```

**En Windows (Command Prompt):**
```batch
.venv\Scripts\activate.bat
```

**En Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Su impulso terminal cambiará para indicar que el entorno virtual está activo.

Para desactivarlo, simplemente ejecute:
```bash
deactivate
```

### Eliminación de un entorno

Un entorno virtual `uv` es sólo un directorio. Para eliminarlo, puede eliminar el directorio.

```bash
# Make sure you are not inside the environment
deactivate

# Remove the directory
rm -rf .venv
```

## 3. Iniciación de un proyecto de pitón

`uv` utiliza `pyproject.toml` para administrar dependencias, siguiendo los estándares PEP 621. Este es el enfoque recomendado sobre `requirements.txt`.

### Iniciar un nuevo proyecto

Para crear un nuevo proyecto Python con un `pyproject.toml`:

```bash
uv init my-project
cd my-project
```

Esto crea una estructura básica de proyecto con un archivo `pyproject.toml`.

### Iniciar en un directorio existente

Si ya tiene un directorio de proyecto:

```bash
uv init
```

Esto creará un `pyproject.toml` en el directorio actual.

## 4. Gestión de paquetes con `pyproject.toml`

`uv` gestiona automáticamente su entorno y dependencias virtuales cuando utiliza `uv add` y `uv remove`.

### Agregar paquetes

Utilice `uv add` para instalar paquetes y actualizar automáticamente su `pyproject.toml`:

```bash
# Add packages to your project
uv add numpy pandas scikit-learn

# Add a specific version
uv add "numpy==1.26.0"

# Add development dependencies
uv add --dev pytest black
```

Este comando:
- Crea/activa automáticamente el entorno virtual
- Instala los paquetes
- Actualizaciones `pyproject.toml` con las dependencias
- Crea/actúa `uv.lock` para construcciones reproducibles

### Eliminar paquetes

Para eliminar un paquete:

```bash
uv remove scikit-learn
```

### Lista de paquetes instalados

Para ver todos los paquetes en su proyecto:

```bash
uv pip list
```

### Dependencias sincronómicas

Para asegurar que su entorno coincida con `pyproject.toml`:

```bash
uv sync
```

## 5. Python corriendo con uv

Puede ejecutar scripts y comandos Python sin activar manualmente el entorno virtual:

```bash
# Run a Python script
uv run python script.py

# Run a module
uv run python -m pytest

# Run an installed tool
uv run black .
```

`uv run` utiliza automáticamente el entorno virtual del proyecto.

## 6. Proyectos compartidos

Al compartir su proyecto, incluya tanto `pyproject.toml` como `uv.lock` en el control de versiones.

### Cerrar y configurar un proyecto

Cualquier persona puede replicar su entorno con:

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install all dependencies (creates venv automatically)
uv sync
```

### Trabajando con Legacy `requirements.txt`

Si necesita trabajar con los archivos `requirements.txt` existentes:

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Generate requirements.txt from current environment
uv pip freeze > requirements.txt
```

Sin embargo, migrar a `pyproject.toml` es recomendable para una mejor gestión de dependencia.

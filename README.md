# Andrés Alfaro
# RAG GUI 
# Recuperación Aumentada con Generación RAG

![Interfaz principal de RAG GUI](https://raw.githubusercontent.com/cafemigao/rag_gui/main/rag_gui-.png)

Este proyecto consiste en una aplicación de interfaz gráfica desarrollada con **Tkinter** para implementar Recuperación Aumentada con Generación (RAG) que trabaja en local, utilizando el modelo de lenguaje **Mistral-7B** en **Python 3**. Aunque se probaron modelos más pequeños, estos generaban respuestas con excesiva "alucinación". Optimizando los parámetros del modelo Mistral-7B, logré respuestas más precisas y una ejecución más eficiente, aunque eso dependerá del computador donde trabaje. Los valores recomendados por defecto los he dejado en:

- `chunk_size=1000`, `chunk_overlap=100`: Para el fraccionamiento del texto.
- `max_new_tokens=500`: Para limitar la longitud de las respuestas generadas.

Con esta configuración, si tienes un equipo decente funcionará, aunque te hará falta un poco de paciencia ya que el modelo es grande y mi ordenador no es precisamente un cohete. Pero si ajustas estos valores a tu gusto, puedes sacarle más jugo, tanto en precisión como en velocidad.


---

## Características
- Carga de documentos en formato HTML, TXT, PDF, DOCX, XLSX, LOGS.
- Procesamiento de documentos para extraer y fragmentar su contenido.
- Indexación de texto mediante FAISS y embeddings de HuggingFace.
- Generación de respuestas a preguntas sobre los documentos cargados.
- Lectura en voz alta de las respuestas generadas.
- Interfaz sencilla e intuitiva basada en Tkinter.

![Procesamiento de un archivo de log](https://raw.githubusercontent.com/cafemigao/rag_gui/main/rag_gui-log.png)

---

## Requisitos del sistema
El software ha sido testeado y funciona correctamente en un equipo con las siguientes características:
- **Sistema operativo:** Ubuntu 6.14.0-13-generic - SMP PREEMPT_DYNAMIC - GNU/Linux con Python 3.8+
- **Procesador:** 12 núcleos CPU
- **Memoria RAM:** 64GB
- **Almacenamiento:** 2TB de disco duro
- **Aceleración por hardware:** Soporte para CUDA opcional para mejorar el rendimiento con GPU.

---

![Interfaz principal de RAG GUI](https://raw.githubusercontent.com/cafemigao/rag_gui/main/rag_gui-respuesta.png)

## Instalación y Dependencias
Es necesario instalar las dependencias requeridas y una cuenta en [huggingface.co](https://huggingface.co/). Comprobar en -> *Gated Repositories* el modelo "accepted" y un Token de HuggingFace en -> *Access Tokens*. Para ejecutar este proyecto, puedes hacer esto con los siguientes comandos:

### 1. Clonar el repositorio
```sh
git clone https://github.com/cafemigao/rag_gui.git
cd rag_gui

### 2. Crear un entorno virtual (opcional pero recomendado)
```sh
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate     # En Windows
```

### 3. Instalar dependencias
```sh
pip install -r requirements.txt
```

Si deseas usar aceleración por GPU, asegúrate de tener los drivers CUDA instalados y que PyTorch los reconozca:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Ejecución del programa
Para iniciar la interfaz, ejecuta el siguiente comando:
```sh
python3 rag_gui.py
```

---

## Archivo `requirements.txt`
Asegúrate de incluir un archivo `requirements.txt` con las siguientes dependencias:
```
torch
tk
pymupdf
python-docx
openpyxl
requests
langchain
langchain-community
langchain-huggingface
transformers
sentence-transformers
faiss-cpu
gtts
```
---

## Notas adicionales
- Si experimentas errores con la generación de voz, asegúrate de tener instalado `mpg123` para reproducir audio.
- Puedes modificar la variable `MODEL_NAME` en el script para probar otros modelos de lenguaje.

---

## Autor
**Andrés Alfaro** - Desarrollo y Pruebas de Sistemas.

---

## Licencia
Este proyecto está bajo la licencia GPL v3 . Puedes ver más detalles en el archivo `LICENSE.md`.



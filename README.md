# Inteligencia Artificial sobre ciberseguridad. 
# Andrés Alfaro.
# Proyecto RAG GUI 
# Interfaz Gráfica para Recuperación Aumentada con Generación

Este proyecto es una aplicación de interfaz gráfica basada en Tkinter para realizar Recuperación Aumentada con Generación (RAG) utilizando un modelo de lenguaje basado en Mistral-7B.

Aunque he probado con otros modelos más pequeños, las respuestas alucinaban en demasía. Conservando este modelo y cambiando algunos parámetros de valores, podrás conseguir mejores respuestas y mayor rápidez de ejecución en:

(chunk_size=1000, chunk_overlap=100)
max_new_tokens=500 

Con estos valores actuales en un computador de los requisitos del sistemas que se definen, deberían tener un poco de paciencia al tratarse de un modelo 'grande' para el computador donde se testea.

---

## Características
- Carga de documentos en formato TXT, PDF, DOCX y XLSX.
- Procesamiento de documentos para extraer y fragmentar su contenido.
- Indexación de texto mediante FAISS y embeddings de HuggingFace 
- Generación de respuestas a preguntas sobre los documentos cargados.
- Lectura en voz alta de las respuestas generadas.
- Interfaz sencilla e intuitiva basada en Tkinter.

---

## Requisitos del sistema
El software ha sido testeado y funciona correctamente en un equipo con las siguientes características:
- **Sistema operativo:** Ubuntu 6.14.0-13-generic - SMP PREEMPT_DYNAMIC - GNU/Linux con Python 3.8+
- **Procesador:** 12 núcleos CPU
- **Memoria RAM:** 64GB
- **Almacenamiento:** 2TB de disco duro
- **Aceleración por hardware:** Soporte para CUDA opcional para mejorar el rendimiento con GPU.

---

## Instalación y Dependencias
Es necesario instalar las dependencias requeridas y una cuenta en huggingface.co. Comprobar en -> Gated Repositories el modelo "accepted" y un Token de HuggingFace en -> Access Tokens. Para ejecutar este proyecto, puedes hacer esto con los siguientes comandos:

### 1. Clonar el repositorio
```sh
 git clone https://github.com/cafemigao/rag_gui.git
 cd rag_gui
```

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
python rag_gui.py
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
Este proyecto está bajo la licencia MIT. Puedes ver más detalles en el archivo `LICENSE`.



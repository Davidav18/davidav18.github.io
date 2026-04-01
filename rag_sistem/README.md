# Proyecto RAG Experimental con LangChain + DeepSeek

## Descripción general

Este proyecto es una **base experimental de un sistema RAG (Retrieval-Augmented Generation)** construida en Python con LangChain, Chroma, embeddings de Hugging Face y un modelo de DeepSeek.

La idea principal del proyecto es cargar contenido desde una fuente web, dividirlo en fragmentos, vectorizarlo y luego responder preguntas usando recuperación semántica y generación con LLM.

Además de la base RAG clásica, el código incluye pruebas o primeras implementaciones de varias estrategias avanzadas de mejora de recuperación y razonamiento, como:

- **Query Decomposition**
- **Step-Back Prompting**
- **HyDE (Hypothetical Document Embeddings)**
- **Routing entre vectorstore y respuesta general**

> **Estado actual:** el proyecto está **funcional como prototipo**, pero todavía está **incompleto**. Varias técnicas ya están definidas en el código, aunque no todas están integradas en el flujo principal de ejecución.

---

## Objetivo del proyecto

Este repositorio sirve como:

1. **Base de aprendizaje** para entender la arquitectura de un sistema RAG.
2. **Sandbox de experimentación** para probar distintas técnicas de recuperación y transformación de consultas.
3. **Punto de partida** para convertirlo más adelante en un sistema RAG más robusto y modular.

---

## Tecnologías utilizadas

- **Python**
- **LangChain**
- **DeepSeek Chat** vía API compatible con OpenAI
- **Chroma** como vector store
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **BeautifulSoup / WebBaseLoader** para ingestión desde web
- **python-dotenv** para manejo de variables de entorno

---

## Fuente de conocimiento actual

Actualmente, el sistema indexa una sola fuente web:

- `https://lilianweng.github.io/posts/2023-06-23-agent/`

Esto significa que el vector store se construye únicamente con el contenido de ese artículo, por lo que las respuestas RAG están limitadas a ese dominio temático.

---

## Arquitectura actual

El proyecto está implementado en un único archivo (`app.py`) y se puede dividir en las siguientes etapas:

### 1. Configuración del entorno

Se cargan variables de entorno con `load_dotenv()` y se inicializa el modelo LLM usando DeepSeek:

- modelo: `deepseek-chat`
- `temperature=0`
- autenticación mediante `DEEPSEEK_API_KEY`

### 2. Ingesta de documentos

Se usa `WebBaseLoader` para descargar el contenido de una página web y filtrar las secciones relevantes del HTML.

### 3. División en chunks

El texto se divide con `RecursiveCharacterTextSplitter` usando:

- `chunk_size = 1000`
- `chunk_overlap = 200`

Esto ayuda a crear fragmentos manejables para embedding y recuperación.

### 4. Embeddings e indexación

Cada fragmento se transforma en embedding usando el modelo:

- `sentence-transformers/all-MiniLM-L6-v2`

Después, los fragmentos se almacenan en **Chroma** para habilitar recuperación semántica.

### 5. Retriever

Se crea un `retriever` a partir del vector store para recuperar los documentos más relevantes frente a una pregunta.

### 6. Prompt base de RAG

El sistema define un prompt simple que obliga al modelo a responder únicamente usando el contexto recuperado.

---

## Técnicas implementadas o preparadas

### A. Query Decomposition

El proyecto incluye una cadena que toma una pregunta y genera **3 subconsultas** relacionadas.

**Propósito:** descomponer una pregunta compleja en preguntas más pequeñas que puedan mejorar la recuperación de información.

**Estado:** implementado en el código, pero **no conectado al flujo principal** de `main`.

---

### B. Step-Back Prompting

Se incluye una lógica de *step-back prompting* basada en few-shot examples. La idea es reformular una pregunta específica en una versión más general para recuperar contexto de apoyo.

Luego se arma un pipeline que combina:

- contexto recuperado desde la pregunta general (*step-back context*)
- contexto recuperado desde la pregunta original
- respuesta final generada con ambos contextos

**Estado:** la cadena final está construida, pero **no se usa actualmente en la ejecución principal**.

---

### C. HyDE

También se implementa una versión de **HyDE**. En este enfoque:

1. el modelo genera un pasaje hipotético que podría responder la pregunta,
2. ese pasaje generado se usa como entrada para la recuperación,
3. los documentos recuperados sirven como contexto para la respuesta final.

**Estado:** esta es la técnica que **sí se usa actualmente** cuando el router decide enviar la consulta al vector store.

---

### D. Routing

El proyecto define un router estructurado con Pydantic y `JsonOutputParser` para decidir entre dos rutas:

- `vectorstore`
- `web_search`

La intención es enviar preguntas relacionadas con los documentos indexados al sistema RAG y enviar el resto a otra fuente.

**Estado real actual:**

- La ruta `vectorstore` sí está conectada al flujo principal.
- La ruta `web_search` **todavía no realiza una búsqueda web real**; actualmente solo hace una respuesta general con el LLM.

Por eso, el nombre `web_search` debe entenderse por ahora como una **ruta pendiente de completar**.

---

## Flujo de ejecución actual

Cuando se ejecuta `app.py`, el sistema sigue este proceso:

1. carga el contenido web;
2. divide el texto en chunks;
3. crea embeddings;
4. construye el vector store en Chroma;
5. recibe una pregunta de prueba;
6. el router decide si la pregunta va a `vectorstore` o `web_search`;
7. si va a `vectorstore`, usa **HyDE + RAG** para responder;
8. si va a `web_search`, responde directamente con el LLM sin búsqueda real.

---

## Qué hace bien este proyecto ahora mismo

- Construye una base RAG funcional desde cero.
- Usa embeddings locales conocidos y ligeros.
- Demuestra varias técnicas avanzadas de mejora de recuperación.
- Tiene una estructura comprensible para seguir aprendiendo o iterando.
- Sirve como prototipo muy bueno para convertirlo en un sistema modular más serio.

---

## Limitaciones actuales

Este proyecto todavía tiene varias limitaciones importantes:

### 1. Solo indexa una única fuente web

Todo el conocimiento del vector store proviene de una sola URL.

### 2. No hay persistencia del vector store

Chroma se crea en memoria durante la ejecución. Si se reinicia el script, se vuelve a procesar todo desde cero.

### 3. El router promete más de lo que ejecuta

Existe una ruta llamada `web_search`, pero todavía no integra una búsqueda real con APIs, navegador o herramienta externa.

### 4. Varias cadenas avanzadas no están integradas

`Query Decomposition` y `Step-Back Prompting` están implementadas, pero no forman parte del flujo principal actual.

### 5. El proyecto está en un solo archivo

Toda la lógica está concentrada en `app.py`, lo que dificulta mantenimiento, pruebas y escalabilidad.

### 6. No hay evaluación ni métricas

No existe todavía una capa para medir calidad de recuperación, precisión, grounding o desempeño del pipeline.

### 7. No hay manejo de errores ni configuración avanzada

Faltan validaciones, logging, persistencia, control de parámetros y separación entre entorno de desarrollo y producción.

---

## Posibles mejoras futuras

Este proyecto puede crecer bastante si se trabajan estos siguientes pasos:

### Corto plazo

- Separar el código en módulos (`ingestion.py`, `retrieval.py`, `routing.py`, `chains.py`, etc.).
- Agregar un archivo `requirements.txt` o `pyproject.toml`.
- Crear persistencia para Chroma.
- Convertir la pregunta de prueba en entrada dinámica por CLI o API.
- Documentar instalación y ejecución paso a paso.

### Mediano plazo

- Implementar búsqueda web real para la ruta `web_search`.
- Integrar `Query Decomposition` y `Step-Back` como estrategias opcionales.
- Agregar re-ranking.
- Permitir múltiples fuentes de datos (PDFs, páginas web, archivos locales, Notion, etc.).
- Añadir trazabilidad y observabilidad del pipeline.

### Largo plazo

- Exponer el proyecto mediante FastAPI o Streamlit.
- Añadir evaluación automática del sistema RAG.
- Incorporar caching, guardrails y tests.
- Crear una versión más productiva orientada a casos reales.

---

## Estructura actual del código

```bash
.
├── app.py
└── .env
```

### Variable de entorno requerida

```env
DEEPSEEK_API_KEY=tu_api_key_aqui
```

---

## Cómo ejecutar

### 1. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

Ejemplo aproximado:

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb sentence-transformers beautifulsoup4 python-dotenv pydantic
```

### 3. Configurar variables de entorno

Crear archivo `.env`:

```env
DEEPSEEK_API_KEY=tu_api_key_aqui
```

### 4. Ejecutar

```bash
python app.py
```

---

## Resumen técnico rápido

En su estado actual, este proyecto:

- **sí implementa un pipeline RAG base funcional**;
- **sí prueba una estrategia avanzada real (HyDE)**;
- **sí define otras estrategias avanzadas para evolución futura**;
- **todavía no integra todas sus piezas en un flujo completo**;
- **todavía no está listo para producción**, pero sí es una base excelente para aprendizaje, iteración y documentación técnica.

---

## Referencia e inspiración

Este proyecto fue construido siguiendo y adaptando ideas del artículo:

**“Building the Entire RAG Ecosystem and Optimizing Every Component”**

Se usó como guía para explorar conceptos como routing, transformación de consultas, indexing y retrieval dentro de un sistema RAG más amplio.

---

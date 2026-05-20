# 🏭 Sistema Inteligente de Detección de EPP
### Despliegue de Modelos de Visión Artificial con Streamlit · UNAB 2026

> **Proyecto educativo de Machine Learning aplicado:** aprende el ciclo completo de un proyecto de visión artificial, desde la recolección y etiquetado de datos hasta el despliegue de una aplicación web pública, sin que el usuario final vea una sola línea de código.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-violet?logo=roboflow)
![Colab](https://img.shields.io/badge/Entrenamiento-Google%20Colab-yellow?logo=googlecolab)

---

## 📋 Tabla de contenidos

1. [¿Qué aprenderás?](#-qué-aprenderás)
2. [¿Qué es Streamlit?](#-qué-es-streamlit-y-por-qué-usarlo-en-ml)
3. [¿Qué es el deployment?](#-qué-es-el-deployment-y-por-qué-importa)
4. [Arquitectura del proyecto](#-arquitectura-del-proyecto)
5. [Estructura del repositorio](#-estructura-del-repositorio)
6. [Parte 1 — Dataset con Roboflow](#-parte-1--dataset-con-roboflow)
7. [Parte 2 — Entrenamiento en Google Colab](#-parte-2--entrenamiento-en-google-colab)
8. [Parte 3 — Publicar en GitHub](#-parte-3--publicar-en-github)
9. [Parte 4 — Deployment en Streamlit Cloud](#-parte-4--deployment-en-streamlit-cloud)
10. [Cómo usar la aplicación](#-cómo-usar-la-aplicación)
11. [¿Cómo funciona el código?](#-cómo-funciona-el-código-app-explicada)
12. [Créditos](#-créditos)

---

## 🎯 ¿Qué aprenderás?

Este proyecto cubre el **ciclo completo de un proyecto real de visión artificial**:

| Etapa | Herramienta | Concepto clave |
|---|---|---|
| **Recolección y etiquetado** | Roboflow | Dataset, anotaciones, clases |
| **Entrenamiento** | Google Colab + YOLOv8 | Transfer learning, épocas, métricas |
| **Exportación del modelo** | Ultralytics `.pt` | Pesos entrenados, artefacto de ML |
| **Control de versiones** | GitHub + Git LFS | Repositorio, archivos binarios grandes |
| **Despliegue web** | Streamlit Cloud | Deployment, interfaz para usuario final |

> **Mensaje clave para los estudiantes:** el objetivo del deployment es que el **usuario final use el modelo sin ver código**. Un modelo guardado en Colab no sirve de nada si nadie más puede acceder a él.

---

## 🌐 ¿Qué es Streamlit y por qué usarlo en ML?

**Streamlit** es una librería de Python que convierte scripts normales en **aplicaciones web interactivas**, sin necesitar conocimientos de HTML, CSS ni JavaScript.

### ¿Por qué es ideal para proyectos de Machine Learning?

```
Científico de datos                    Usuario final
    escribe Python     →   Streamlit   →   ve una app web
    (modelo, lógica)       (convierte)     (sin ver código)
```

### Comparación con otras opciones

| Herramienta | Requiere | Curva de aprendizaje | Ideal para |
|---|---|---|---|
| **Streamlit** | Solo Python | ⭐ Baja | Prototipos ML, demos |
| Flask / Django | Python + HTML + JS | ⭐⭐⭐ Alta | Aplicaciones en producción |
| Gradio | Solo Python | ⭐ Baja | Demos de modelos simples |
| Power BI | Ninguno | ⭐⭐ Media | Dashboards de datos |

### Componentes de Streamlit usados en este proyecto

```python
st.set_page_config(...)    # Configuración de la página
st.title(...)              # Títulos y subtítulos
st.image(...)              # Mostrar imágenes
st.file_uploader(...)      # Subir archivos
st.camera_input(...)       # Acceder a la cámara web
st.spinner(...)            # Indicador de carga
st.success / st.error(...) # Alertas de semáforo
st.dataframe(...)          # Tablas interactivas
st.columns(...)            # Layout en columnas
@st.cache_resource         # Caché del modelo (no lo carga dos veces)
```

> **Streamlit Community Cloud** permite desplegar estas apps de forma **gratuita** vinculando directamente con GitHub. Cada `git push` actualiza la app automáticamente.

---

## 🚀 ¿Qué es el Deployment y por qué importa?

**Deployment** (despliegue) es el proceso de llevar un modelo entrenado a un entorno donde **usuarios reales** puedan usarlo.

### El problema sin deployment

```
[Científico]  →  modelo en Colab  →  ¿Cómo lo usa el operario de planta?
                                              ❌ No puede.
```

### La solución con deployment

```
[Científico]  →  GitHub  →  Streamlit Cloud  →  URL pública
                                                      ↓
                                              [Operario de planta]
                                              sube foto → ve resultado
                                              (sin saber que hay Python)
```

### Ciclo completo de este proyecto

```
Roboflow          Colab              GitHub           Streamlit Cloud
(dataset)  →  (entrenar YOLO)  →  (subir modelo)  →  (app pública)
   ↓               ↓                   ↓                    ↓
Imágenes       best.pt             Repositorio        URL accesible
etiquetadas    entrenado           versionado         desde cualquier
con EPP        con EPP             + código           navegador
```

---

## 🏗️ Arquitectura del proyecto

```
📷 Imagen de entrada (trabajador)
         ↓
┌─────────────────────────────────┐
│  MODELO 1: YOLOv8n (general)    │  ← Detecta PERSONAS en la imagen
│  yolov8n.pt (preentrenado)      │    Clase ID 0 = persona
└─────────────┬───────────────────┘
              │ Recorta cada persona detectada
              ↓
┌─────────────────────────────────┐
│  MODELO 2: YOLOv8 (custom)      │  ← Detecta EPP en cada persona
│  best.pt (entrenado en Colab)   │    Clases: casco, chaleco, guantes...
└─────────────┬───────────────────┘
              │
              ↓
       ¿Tiene Casco Y Chaleco?
         /              \
        Sí               No
        ↓                ↓
   🟢 ACCESO         🔴 ACCESO
   PERMITIDO         DENEGADO
```

> **¿Por qué dos modelos?** El modelo personalizado (`best.pt`) fue entrenado con imágenes de EPP. Usar primero un detector general de personas mejora la precisión, ya que el modelo analiza cada trabajador de forma individual en lugar de buscar EPP en toda la imagen.

---

## 📁 Estructura del repositorio

```
📦 repositorio/
│
├── 🐍 app.py                    ← Aplicación Streamlit principal
├── 🤖 best.pt                   ← Modelo YOLOv8 entrenado (EPP) — Git LFS
├── 🤖 yolov8n.pt                ← Modelo YOLOv8n base (personas)
├── 🖼️  logo.jpg                  ← Logo del proyecto
│
├── 📄 requirements.txt          ← Dependencias Python
├── 📄 packages.txt              ← Paquetes del sistema (libGL para OpenCV)
├── 📄 runtime.txt               ← Versión de Python para Streamlit Cloud
├── 📄 .gitattributes            ← Configuración Git LFS para archivos .pt
│
├── 📓 notebook/
│   └── readme.md                ← Enlace al cuaderno de entrenamiento en Colab
│
├── 🖼️  FOTOS/                    ← Imágenes de prueba para la app
│
└── 📄 README.md                 ← Este archivo
```

### Archivos de configuración explicados

**`requirements.txt`** — librerías Python que Streamlit Cloud instala automáticamente:
```
ultralytics
streamlit
pillow
numpy
pandas
```

**`packages.txt`** — paquetes del sistema operativo necesarios para OpenCV:
```
libgl1
libglib2.0-0
```

**`runtime.txt`** — fija la versión de Python en Streamlit Cloud:
```
python-3.10
```

**`.gitattributes`** — indica a Git LFS que los archivos `.pt` son binarios grandes:
```
*.pt filter=lfs diff=lfs merge=lfs -text
```

---

## 📸 Parte 1 — Dataset con Roboflow

### ¿Qué es Roboflow?

**Roboflow** es una plataforma web para gestionar datasets de visión artificial. Permite:
- **Subir y organizar** imágenes
- **Etiquetar** objetos dibujando bounding boxes
- **Aumentar** el dataset automáticamente (flip, rotación, brillo)
- **Exportar** en el formato que YOLO necesita

### Dataset usado en este proyecto

El dataset de **PPE Detection** fue obtenido del workspace `cicatriz` en Roboflow, con las siguientes clases etiquetadas:

| Clase (inglés) | Clase (español) | Descripción |
|---|---|---|
| `helmet` | Casco | Casco de seguridad industrial |
| `vest` | Chaleco | Chaleco reflectivo |
| `gloves` | Guantes | Guantes de protección |
| `glasses` | Gafas | Gafas de seguridad |
| `boots` | Botas | Botas industriales |
| `earmuffs` | Orejeras | Protección auditiva |
| `person` | Persona | Trabajador (omitido en el resultado final) |

### Cómo se descargó el dataset en Colab

```python
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="dSMfDD4uPaMCKEoGOP5q")
project = rf.workspace("cicatriz").project("PPE Detection")
dataset = project.version(1).download("yolov8")
```

> El formato `"yolov8"` exporta las imágenes con archivos `.txt` de anotaciones. Cada línea del `.txt` contiene `clase x_centro y_centro ancho alto` en coordenadas normalizadas (valores entre 0 y 1).

### ¿Cómo se etiquetan las imágenes en Roboflow?

1. Se sube la imagen a Roboflow
2. Se dibuja un rectángulo (bounding box) alrededor de cada objeto
3. Se asigna la clase (`helmet`, `vest`, etc.)
4. Roboflow genera el archivo `.txt` de anotación automáticamente

```
Ejemplo de anotación YOLOv8 para una imagen con casco y chaleco:

0  0.512 0.234 0.180 0.320   ← persona (clase 0)
4  0.515 0.198 0.162 0.089   ← casco   (clase 4)
6  0.510 0.350 0.210 0.280   ← chaleco (clase 6)

Formato: clase  x_centro  y_centro  ancho  alto
```

---

## 🧠 Parte 2 — Entrenamiento en Google Colab

📓 **Cuaderno de entrenamiento completo:**
[**Abrir en Google Colab →**](https://colab.research.google.com/drive/1rhCXLT5rOQns6BWy7iZUiTupvggh5Ghp?usp=sharing)

### ¿Qué es YOLOv8?

**YOLO** (*You Only Look Once*) es una familia de modelos de detección de objetos en tiempo real. La versión 8, desarrollada por **Ultralytics**, es la usada en este proyecto por su balance entre velocidad y precisión.

### Transfer Learning — por qué no entrenamos desde cero

Entrenar una red neuronal desde cero para visión artificial requiere millones de imágenes y semanas de cómputo. En cambio, usamos **transfer learning**:

```
YOLOv8 preentrenado          Dataset EPP (Roboflow)
(aprendió con millones   +   (nuestras imágenes
 de imágenes de COCO)         de trabajadores)
         ↓                           ↓
    Pesos base              Fine-tuning (ajuste fino)
    (ya sabe ver)           (aprende EPP específico)
              ↘             ↙
               best.pt final
```

### Pasos del entrenamiento en Colab

**Paso 1 — Activar GPU** (obligatorio para velocidad):
`Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`

**Paso 2 — Instalar Ultralytics:**
```python
!pip install ultralytics roboflow
```

**Paso 3 — Descargar dataset desde Roboflow:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="dSMfDD4uPaMCKEoGOP5q")
project = rf.workspace("cicatriz").project("PPE Detection")
dataset = project.version(1).download("yolov8")
```

**Paso 4 — Entrenar el modelo:**
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # Carga el modelo base preentrenado

results = model.train(
    data="PPE Detection-1/data.yaml",  # Configuración del dataset
    epochs=50,                          # Número de épocas
    imgsz=640,                          # Tamaño de imagen
    batch=16,                           # Imágenes por batch
    project="ppe_detector",             # Carpeta de resultados
    name="entrenamiento_v1"
)
```

**Paso 5 — Evaluar el modelo:**
```python
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

**Paso 6 — Descargar `best.pt`:**
```python
from google.colab import files
files.download("ppe_detector/entrenamiento_v1/weights/best.pt")
```

> **`best.pt`** contiene los pesos del modelo que obtuvo el mejor resultado durante el entrenamiento. Es el único archivo que necesita la app Streamlit para hacer predicciones.

### ¿Qué significan las métricas?

| Métrica | Significado | Valor esperado |
|---|---|---|
| **mAP50** | Precisión con umbral de overlap 50% | > 0.80 |
| **Precisión** | De lo que predice, ¿cuánto es correcto? | > 0.80 |
| **Recall** | De lo que existe, ¿cuánto detecta? | > 0.75 |
| **Loss** | Error del modelo | Debe bajar con las épocas |

### ¿Puedo continuar el entrenamiento sin reiniciar?

**Sí.** Mientras el entorno de Colab no se reinicie, los pesos permanecen en memoria. Puedes volver a ejecutar solo la celda del bucle de entrenamiento con más épocas y el modelo continuará aprendiendo desde donde quedó. Solo **no re-ejecutes** las celdas que construyen el modelo o los optimizadores, ya que eso reiniciaría los pesos desde cero.

---

## 📦 Parte 3 — Publicar en GitHub

### ¿Por qué GitHub y no solo Drive?

GitHub permite versionar el código, vincularse directamente con Streamlit Cloud, colaborar en equipo y automatizar el redespliegue al hacer `git push`.

### Paso 3.1 — Crear el repositorio

1. Ve a [github.com](https://github.com) e inicia sesión
2. Haz clic en **`+` → `New repository`**
3. Configura:
   - **Nombre:** `ppe-detector-unab`
   - **Visibilidad:** `Public` ← necesario para Streamlit Cloud gratuito
   - Activa **`Add a README file`**
4. Haz clic en **`Create repository`**

### Paso 3.2 — Configurar Git LFS para archivos `.pt`

Los modelos `.pt` de YOLO pesan entre 6 MB y 150 MB. GitHub bloquea archivos mayores a 100 MB sin Git LFS.

```bash
# Instalar Git LFS (solo una vez por máquina)
git lfs install

# Dentro de la carpeta del proyecto:
git lfs track "*.pt"
git add .gitattributes
```

### Paso 3.3 — Clonar y subir todos los archivos

```bash
git clone https://github.com/TU-USUARIO/ppe-detector-unab.git
cd ppe-detector-unab
```

Copia a esta carpeta: `app.py`, `best.pt`, `yolov8n.pt`, `requirements.txt`, `packages.txt`, `runtime.txt`, `logo.jpg`.

```bash
git add .
git commit -m "feat: app detección EPP con YOLOv8 + Streamlit"
git push origin main
```

### Paso 3.4 — Verificar en GitHub

Confirma que los archivos `.pt` aparecen con la etiqueta **`Stored with Git LFS`** en la vista del repositorio.

---

## 🚀 Parte 4 — Deployment en Streamlit Cloud

### Paso 4.1 — Crear cuenta

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **`Sign up`**
3. Selecciona **`Continue with GitHub`**
4. Autoriza el acceso y completa el registro

### Paso 4.2 — Crear la aplicación

1. En el dashboard, haz clic en **`New app`**
2. Selecciona **`From existing repo`**
3. Completa los campos:

| Campo | Valor |
|---|---|
| **Repository** | `TU-USUARIO/ppe-detector-unab` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL** | `ppe-detector-unab` (personalizable) |

4. Haz clic en **`Deploy!`**

### Paso 4.3 — Proceso de despliegue

Streamlit instalará todo automáticamente leyendo los archivos de configuración:

```
📄 packages.txt     →  instala libGL (necesario para OpenCV/YOLO en Linux)
📄 runtime.txt      →  usa Python 3.10
📄 requirements.txt →  instala ultralytics, streamlit, pillow, pandas, numpy
🤖 best.pt          →  descargado desde Git LFS
```

El primer despliegue puede tardar **5–10 minutos** por la instalación de Ultralytics.

### Paso 4.4 — Tu app está en vivo

```
https://ppe-detector-unab.streamlit.app
```

Comparte esta URL. El usuario final **solo necesita un navegador** — no instala nada.

### Paso 4.5 — Ciclo de actualización

```bash
# Editas el código → subes a GitHub → Streamlit redesplieg automáticamente
git add app.py
git commit -m "mejora: actualizo lógica del semáforo"
git push origin main
# → La app se actualiza en ~2 minutos sin hacer nada más
```

---

## 🎮 Cómo usar la aplicación

### Métodos de entrada

| Método | Cuándo usarlo |
|---|---|
| **📁 Subir imagen** | Fotos ya tomadas (JPG, PNG, JPEG) |
| **📷 Cámara web** | Captura en tiempo real desde el navegador |

### Flujo de uso

1. Selecciona el método de entrada
2. Sube o captura la foto del trabajador
3. La app detecta automáticamente las personas presentes
4. Para cada persona muestra:
   - Recorte con bounding boxes de los EPP detectados
   - **🚥 Semáforo de acceso** (verde o rojo)
   - **📊 Tabla de analítica** con cada EPP y su nivel de confianza

### Lógica del semáforo

| Resultado | Condición | Mensaje |
|---|---|---|
| 🟢 **ACCESO PERMITIDO** | Detecta **Casco** Y **Chaleco** | Cumple requisitos mínimos |
| 🔴 **ACCESO DENEGADO** | Falta **Casco** O **Chaleco** | Indica exactamente cuál falta |

> Los EPP opcionales (guantes, gafas, botas, orejeras) se muestran en la tabla de analítica pero no afectan el semáforo.

---

## 💻 ¿Cómo funciona el código? (app explicada)

### Estructura general de `app.py`

```python
# 1. Configuración de página — SIEMPRE debe ser la primera línea de Streamlit
st.set_page_config(...)

# 2. Diccionario de traducción inglés → español
TRADUCCION_CLASES = {"helmet": "Casco", "vest": "Chaleco", ...}

# 3. Carga de modelos con caché (se cargan una sola vez)
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")   # Detector general de personas
    modelo_ppe      = YOLO("best.pt")      # Detector EPP personalizado
    return modelo_personas, modelo_ppe

# 4. Interfaz de usuario (entrada de imagen)
foto = st.file_uploader(...)    # o st.camera_input(...)

# 5. Pipeline: detectar personas → recortar → detectar EPP → semáforo → tabla
```

### El decorador `@st.cache_resource` — clave de rendimiento

```python
@st.cache_resource
def load_models():
    ...
```

Sin este decorador, cada vez que el usuario sube una imagen, Streamlit recargaría los modelos desde disco. Con `@st.cache_resource`, los modelos se cargan **una sola vez** al iniciar la app y permanecen en memoria para todas las consultas posteriores.

### Pipeline de detección en dos etapas

```python
# ETAPA 1: Detectar personas con modelo general (COCO, clase 0 = persona)
resultados_personas = modelo_personas(img_np)[0]
for box in resultados_personas.boxes:
    if int(box.cls[0]) == 0:
        personas.append(coordenadas_bounding_box)

# ETAPA 2: Para cada persona recortada, detectar EPP
for x1, y1, x2, y2 in personas:
    persona_crop  = imagen.crop((x1, y1, x2, y2))   # Recortar persona
    resultado_ppe = modelo_ppe(persona_crop)[0]       # Inferencia EPP
```

### Validación de requisitos de seguridad

```python
requeridos = {"Casco", "Chaleco"}       # EPP mínimo obligatorio
presentes  = set(etiquetas_detectadas)   # EPP encontrado en la imagen

if requeridos.issubset(presentes):
    st.success("🟢 ACCESO PERMITIDO")
else:
    faltantes = requeridos - presentes
    st.error(f"🔴 ACCESO DENEGADO: Faltan {faltantes}")
```

---

## 🛠️ Solución de problemas frecuentes

### ❌ `libGL.so.1: cannot open shared object file`
Asegúrate de que `packages.txt` contiene `libgl1` y `libglib2.0-0`. Sin estas librerías del sistema, OpenCV no puede correr en Linux (el sistema operativo de Streamlit Cloud).

### ❌ El archivo `best.pt` no aparece en GitHub o pesa 0 bytes
Verifica que ejecutaste `git lfs install` y que `.gitattributes` contiene la línea `*.pt filter=lfs diff=lfs merge=lfs -text` antes del primer commit del modelo.

### ❌ "No se detectaron personas en la imagen"
Prueba con una imagen donde el trabajador sea visible completamente y con buena iluminación. El detector usa el modelo general YOLOv8n entrenado con el dataset COCO.

### ❌ El modelo detecta EPP con baja confianza
Agrega más imágenes diversas en Roboflow y reentrena el modelo con más épocas. La calidad del dataset es el factor más importante en la precisión final.

### ❌ Streamlit muestra `ModuleNotFoundError: ultralytics`
Verifica que `requirements.txt` esté en la raíz del repositorio y que `ultralytics` esté escrito correctamente (sin comillas ni espacios extra).

---

## 👥 Créditos

| Rol | Detalle |
|---|---|
| **Autor** | Alfredo Díaz · UNAB 2026 |
| **Institución** | Universidad Autónoma de Bucaramanga (UNAB) |
| **Unidad** | Centro de Competencias Digitales (CCD) |
| **Ciudad** | Bucaramanga, Colombia |
| **Modelo base** | YOLOv8n — Ultralytics |
| **Dataset** | PPE Detection — Roboflow (workspace: cicatriz) |
| **Entrenamiento** | Google Colab T4 GPU |
| **Interfaz** | Streamlit Community Cloud |
| **Cuaderno de entrenamiento** | [Ver en Google Colab](https://colab.research.google.com/drive/1rhCXLT5rOQns6BWy7iZUiTupvggh5Ghp?usp=sharing) |

---

> 🏆 **Proyecto educativo de IA aplicada a Seguridad Industrial**  
> *"Del dato etiquetado a la aplicación en producción — sin que el usuario vea una sola línea de código"*  
> CCD · UNAB · 2026

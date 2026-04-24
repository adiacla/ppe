import os
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 1. set_page_config DEBE ser el primer comando de Streamlit
st.set_page_config(page_title="Detección de PPE", layout="wide", page_icon="🏭")

# Evita el warning de Ultralytics en Streamlit Cloud
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -----------------------
# Diccionario de Traducción
# -----------------------
TRADUCCION_CLASES = {
    "boots": "Botas",
    "earmuffs": "Orejeras",
    "glasses": "Gafas",
    "gloves": "Guantes",
    "helmet": "Casco",
    "person": "Persona",
    "vest": "Chaleco"
}

# -----------------------
# Cargar modelos con manejo seguro de errores
# -----------------------
@st.cache_resource
def load_models():
    # Verificamos si los archivos existen antes de intentar cargarlos
    if not os.path.exists("yolov8n.pt"):
        st.error("🛑 Error: No se encontró el archivo 'yolov8n.pt'. Asegúrate de que esté en el repositorio.")
        st.stop() # Detiene la ejecución sin mostrar un traceback feo al usuario
        
    if not os.path.exists("best.pt"):
        st.error("🛑 Error: No se encontró el archivo 'best.pt'. Asegúrate de que esté en el repositorio. Si pesa mucho, usa Git LFS.")
        st.stop()

    try:
        modelo_personas = YOLO("yolov8n.pt")
        modelo_ppe = YOLO("best.pt")
        return modelo_personas, modelo_ppe
    except Exception as e:
        st.error(f"Error interno al cargar los modelos: {e}")
        st.stop()

modelo_personas, modelo_ppe = load_models()

# -----------------------
# UI y Configuración
# -----------------------
st.title("🏭 Sistema Inteligente de Detección de EPP")
st.markdown("""
**📌 Instrucciones de uso:**
1. Sube una fotografía o usa la cámara web.
2. El sistema detectará automáticamente a las personas en la imagen.
3. Se verificará si portan el Equipo de Protección Personal obligatorio (**Casco y Chaleco**).
4. El semáforo indicará si la persona está autorizada para ingresar a la planta.
""")
st.markdown("---")

# -----------------------
# Selección de entrada de imagen (Archivo o Cámara)
# -----------------------
opcion_entrada = st.radio("Selecciona el método de entrada:", ["Subir imagen 📁", "Usar cámara web 📷"])

foto = None
if opcion_entrada == "Subir imagen 📁":
    foto = st.file_uploader("Sube una imagen para analizar", type=["jpg", "png", "jpeg"])
else:
    foto = st.camera_input("Toma una foto del trabajador")

# -----------------------
# Procesamiento de la Imagen
# -----------------------
if foto:
    # Cargar con PIL y asegurar que sea RGB
    imagen_original = Image.open(foto).convert("RGB")
    
    with st.expander("Ver imagen original", expanded=False):
        st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    # Convertir a numpy para que YOLO lo entienda
    img_np = np.array(imagen_original)

    # -----------------------
    # Detectar personas (YOLOv8n)
    # -----------------------
    with st.spinner("Detectando personas..."):
        resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # id 0 es 'persona' en YOLOv8 normal
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    if len(personas) == 0:
        st.warning("⚠️ No se detectaron personas en la imagen.")
    else:
        st.subheader(f"👥 Se han detectado {len(personas)} persona(s) en la imagen")
        st.markdown("---")

    # -----------------------
    # Procesar cada persona con modelo PPE
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):
        st.markdown(f"### 👤 Trabajador {i}")
        
        # Recortar la persona usando PIL
        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)

        # Predecir PPE en el recorte
        resultados_ppe = modelo_ppe(persona_np)[0]

        # Preparar para dibujar cajas con PIL
        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []
        datos_analitica = [] # Para guardar datos para la tabla

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_ppe.names[cls]
            
            # Omitimos la etiqueta "person" del modelo PPE
            if label_ingles == "person":
                continue

            # Traducir al español usando el diccionario
            label_espanol = TRADUCCION_CLASES.get(label_ingles, label_ingles.capitalize())
            conf = float(box.conf[0])
            
            etiquetas.append(label_espanol)
            datos_analitica.append({
                "Equipo Detectado": label_espanol, 
                "Confianza": f"{conf*100:.2f}%",
                "_conf_num": conf # Campo oculto para ordenar fácilmente
            })

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])

            # Dibujar caja de predicción con PIL
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#00FF00", width=3)
            draw.text((x1o, max(0, y1o - 15)), f"{label_espanol} {conf:.2f}", fill="#00FF00")

        # Configurar columnas para UI
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Recorte Trabajador {i}", use_container_width=True)

        with col2:
            # -----------------------
            # Semáforo y Validación PPE
            # -----------------------
            st.markdown("#### 🚥 Control de Acceso a Planta")
            
            # Requisitos OBLIGATORIOS (solo Casco y Chaleco)
            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("🟢 **ACCESO PERMITIDO:** El trabajador cumple con el equipo de seguridad obligatorio (Casco y Chaleco).")
            else:
                faltantes = requeridos - presentes
                st.error(f"🔴 **ACCESO DENEGADO:** Faltan equipos obligatorios: **{', '.join(faltantes)}**")

            # -----------------------
            # Analítica Predictiva
            # -----------------------
            st.markdown("#### 📊 Analítica del Modelo")
            if datos_analitica:
                # Crear DataFrame y ordenar
                df_analitica = pd.DataFrame(datos_analitica)
                df_analitica = df_analitica.sort_values(by="_conf_num", ascending=False).drop(columns=["_conf_num"]).reset_index(drop=True)
                st.dataframe(df_analitica, use_container_width=True)
            else:
                st.warning("⚠️ El modelo no detectó ningún equipo de protección en este trabajador.")
        
        st.markdown("---")

# -----------------------
# Pie de página
# -----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #888888; font-size: 14px;'>"
    "© Alfredo Diaz UNAB 2026. Todos los derechos reservados."
    "</p>", 
    unsafe_allow_html=True
)
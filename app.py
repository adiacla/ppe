import os
# Evita el warning de Ultralytics en Streamlit Cloud
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

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
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    try:
        modelo_personas = YOLO("yolov8n.pt")
    except Exception as e:
        mensaje = "No se pudo cargar yolov8n.pt. Asegúrate de que el archivo exista."
        st.error(mensaje)
        raise RuntimeError(mensaje) from e

    try:
        modelo_ppe = YOLO("best.pt")
    except Exception as e:
        mensaje = "No se pudo cargar best.pt. Asegúrate de que el archivo exista."
        st.error(mensaje)
        raise RuntimeError(mensaje) from e
        
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# UI y Configuración
# -----------------------
st.set_page_config(page_title="Detección de PPE", layout="wide")

st.title("🏭 Sistema Inteligente de Detección de EPP")
st.markdown("""
**📌 Instrucciones de uso:**
1. Sube una fotografía del trabajador o usa la cámara en vivo.
2. El sistema detectará automáticamente a las personas.
3. Se verificará si portan el Equipo de Protección Personal obligatorio (**Casco y Chaleco**).
4. El semáforo indicará si la persona está autorizada para ingresar.
""")
st.markdown("---")

# -----------------------
# Captura de Imagen (Archivo o Cámara)
# -----------------------
col_input1, col_input2 = st.columns(2)

with col_input1:
    foto_subida = st.file_uploader("1️⃣ Sube una imagen para analizar", type=["jpg", "png", "jpeg"])

with col_input2:
    foto_camara = st.camera_input("2️⃣ O toma una foto desde la cámara")

# Prioridad a la cámara si ambas están activas
foto_final = foto_camara if foto_camara is not None else foto_subida

if foto_final:
    # Cargar imagen con PIL
    imagen_original = Image.open(foto_final).convert("RGB")
    
    with st.expander("Ver imagen original", expanded=False):
        st.image(imagen_original, caption="Imagen capturada", use_container_width=True)

    # Convertir a numpy solo para que YOLO lo procese
    img_np = np.array(imagen_original)

    # -----------------------
    # Detectar personas (YOLOv8n)
    # -----------------------
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        # El ID 0 en el modelo COCO de YOLOv8 es 'person'
        if cls == 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    st.subheader(f"👥 Se han detectado {len(personas)} persona(s) en la imagen")
    st.markdown("---")

    # -----------------------
    # Procesar cada persona con modelo PPE
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):
        st.markdown(f"### 👤 Trabajador {i}")
        
        # Recorte usando PIL
        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)

        # Predecir PPE en el recorte
        resultados_ppe = modelo_ppe(persona_np)[0]

        # Crear lienzo PIL para dibujar
        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []
        datos_analitica = [] 

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_ppe.names[cls]
            
            # Omitimos la etiqueta "person" si el modelo PPE la vuelve a detectar
            if label_ingles == "person":
                continue

            label_espanol = TRADUCCION_CLASES.get(label_ingles, label_ingles.capitalize())
            conf = float(box.conf[0])
            
            etiquetas.append(label_espanol)
            datos_analitica.append({
                "Equipo Detectado": label_espanol, 
                "Confianza Raw": conf, 
                "Confianza": f"{conf*100:.2f}%"
            })

            # Coordenadas relativas al recorte
            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])

            # Dibujar caja y texto usando PIL
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#00FF00", width=4)
            draw.text((x1o + 5, max(0, y1o - 15)), f"{label_espanol}", fill="#00FF00")
                
        # Creación de columnas para mostrar resultados por cada trabajador
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Recorte Trabajador {i}", use_container_width=True)

        with col2:
            # -----------------------
            # Semáforo y Validación PPE
            # -----------------------
            st.markdown("#### 🚥 Control de Acceso a Planta")
            
            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("🟢 **ACCESO PERMITIDO:** El trabajador cumple con el equipo de seguridad obligatorio (Casco y Chaleco).")
            else:
                faltantes = requeridos - presentes
                st.error(f"🔴 **ACCESO DENEGADO:** Riesgo crítico de seguridad. Faltan los siguientes equipos obligatorios: **{', '.join(faltantes)}**")

            # -----------------------
            # Analítica Predictiva
            # -----------------------
            st.markdown("#### 📊 Analítica del Modelo")
            if datos_analitica:
                df_analitica = pd.DataFrame(datos_analitica)
                # Ordenar de mayor a menor confianza de forma limpia
                df_analitica = df_analitica.sort_values(by="Confianza Raw", ascending=False)
                # Borramos la columna auxiliar para no ensuciar la vista del usuario
                df_analitica = df_analitica.drop(columns=["Confianza Raw"]).reset_index(drop=True)
                
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
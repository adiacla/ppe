import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# -----------------------
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Sistema PPE", layout="wide")
st.title("🦺 Sistema Inteligente de PPE")

foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if foto:
    imagen_original = Image.open(foto).convert("RGB")
    st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    # Convertir a numpy (YOLO usa esto)
    img_np = np.array(imagen_original)

    # -----------------------
    # Detectar personas
    # -----------------------
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # persona
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    st.subheader(f"👥 Personas detectadas: {len(personas)}")

    # -----------------------
    # Procesar cada persona
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):

        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)

        resultados_ppe = modelo_ppe(persona_np)[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label = modelo_ppe.names[cls]
            conf = float(box.conf[0])
            etiquetas.append(label)

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])

            # Dibujar caja
            draw.rectangle([x1o, y1o, x2o, y2o], outline="green", width=3)
            draw.text((x1o, y1o - 10), f"{label} {conf:.2f}", fill="green")

        # Mostrar resultado
        st.markdown(f"### 👤 Persona {i}")
        st.image(persona_crop, width=300)

        # -----------------------
        # Validación PPE
        # -----------------------
        requeridos = {"Casco", "Chaleco", "Botas"}
        presentes = set(etiquetas)

        if requeridos.issubset(presentes):
            st.success("✅ Cumple con PPE")
        else:
            faltantes = requeridos - presentes
            st.error(f"🚨 Faltan: {', '.join(faltantes)}")

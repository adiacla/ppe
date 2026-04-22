import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# Cargar modelos (cache)
# -----------------------
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# Configuración
# -----------------------
st.set_page_config(page_title="Sistema PPE", layout="wide")

st.title("🦺 Sistema Inteligente de PPE")

# -----------------------
# Input
# -----------------------
foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if foto:
    imagen_original = Image.open(foto)
    st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    # Convertir a OpenCV
    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # -----------------------
    # Detectar personas
    # -----------------------
    resultados_personas = modelo_personas(img_cv)[0]

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

        persona_img = img_cv[y1:y2, x1:x2].copy()

        if persona_img.size == 0:
            continue

        resultados_ppe = modelo_ppe(persona_img)[0]

        etiquetas = []

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label = modelo_ppe.names[cls]
            conf = float(box.conf[0])
            etiquetas.append(label)

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])

            cv2.rectangle(persona_img, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
            cv2.putText(
                persona_img,
                f"{label} {conf:.2f}",
                (x1o, y1o - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Mostrar resultado
        st.markdown(f"### 👤 Persona {i}")
        st.image(persona_img, channels="BGR", width=300)

        # -----------------------
        # Validación PPE
        # -----------------------
        requeridos = {"casco", "chaleco", "botas"}
        presentes = set(etiquetas)

        if requeridos.issubset(presentes):
            st.success("✅ Cumple con PPE")
        else:
            faltantes = requeridos - presentes
            st.error(f"🚨 Faltan: {', '.join(faltantes)}")


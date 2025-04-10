import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile

# Cargar modelos
modelo_personas = YOLO("yolov8n.pt")     # Detección de personas
modelo_ppe = YOLO("best.pt")             # Detección de PPE

# Configuración de la página
st.set_page_config(page_title="Sistema Inteligente de uso de PPE", layout="wide")

# Encabezado con logo y título
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("logo.jpg", width=80)
with col2:
    st.title("Sistema Inteligente de uso de PPE")

# Introducción
st.markdown("""
Bienvenido al **Sistema Inteligente de uso de Equipos de Protección Personal (PPE)**.  
Esta herramienta utiliza visión por computadora para verificar si las personas están utilizando el equipo de protección necesario (casco, chaleco y botas) antes de ingresar a una fábrica.

---  
""")

# Instrucciones
st.subheader("📌 Instrucciones de uso")
st.markdown("""
1. Elige una opción: cargar una imagen o tomar una foto.  
2. Presiona el botón **Enviar Foto**.  
3. El sistema detectará personas y evaluará el uso correcto del equipo de protección personal (PPE).  
""")

# Tabs para seleccionar entre carga y cámara
tab1, tab2 = st.tabs(["📁 Subir Imagen", "📷 Tomar Foto"])

# Variables para imagen y bandera de envío
imagen_original = None
procesar = False

with tab1:
    foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if st.button("📤 Enviar Foto", key="upload"):
        if foto:
            imagen_original = Image.open(foto)
            procesar = True
        else:
            st.warning("Por favor, sube una imagen antes de enviar.")

with tab2:
    captura = st.camera_input("Captura una foto")
    if st.button("📤 Enviar Foto", key="camera"):
        if captura:
            imagen_original = Image.open(captura)
            procesar = True
        else:
            st.warning("Por favor, toma una foto antes de enviar.")

# Procesamiento si hay imagen
if procesar and imagen_original:
    # Mostrar imagen original
    st.subheader("🔍 Imagen cargada")
    st.image(imagen_original, use_container_width=True)

    # Convertir imagen a formato OpenCV
    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detección de personas
    resultados_personas = modelo_personas(img_cv)[0]
    personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]

    st.subheader(f"👥 Personas detectadas: {len(personas_detectadas)}")

    # Evaluar cada persona
    for i, persona in enumerate(personas_detectadas, start=1):
        x1, y1, x2, y2, conf, clase = map(int, persona[:6])
        persona_img = img_cv[y1:y2, x1:x2]

        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, persona_img)

            # Aplicar modelo PPE
            resultados_ppe = modelo_ppe(temp_file.name)[0]
            etiquetas_detectadas = [modelo_ppe.names[int(d.cls)] for d in resultados_ppe.boxes]

            # Mostrar resultados
            st.markdown(f"### 👤 Persona {i}")
            st.image(persona_img, caption="Persona detectada", channels="BGR", width=300)

            st.markdown("**Objetos detectados:** " + ", ".join(etiquetas_detectadas))

            requeridos = {"casco", "chaleco", "botas"}
            presentes = set(etiquetas_detectadas)

            if requeridos.issubset(presentes):
                st.success("✅ Cumple con los requisitos para el ingreso a la fábrica 🏭")
            else:
                faltantes = requeridos - presentes
                st.error(f"🚨 ALERTA: No cumple con los requisitos del PPE. Faltan: {', '.join(faltantes)}")

    st.markdown("---")
    st.markdown("**Autor: Alfredo Díaz**  \nUnab 2025! ©️")

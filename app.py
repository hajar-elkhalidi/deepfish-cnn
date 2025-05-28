import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import requests
import os

# === Configuration ===
st.set_page_config(page_title="DeepFish CNN")
st.title(" Reconnaissance d’espèces de poissons dans des images sous-marines")
st.write("Téléverse une image de poisson pour prédire son espèce 🐟🐠🐡")

# === Chargement du modèle ===
@st.cache_resource
def download_model():
    model_url = "https://github.com/hajar-elkhalidi/deepfish-cnn/releases/download/CNN_model_v0.2/modele_final.h5"
    model_path = "modele_final.h5"

    if not os.path.exists(model_path):
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise Exception(f"Erreur lors du téléchargement du modèle : {response.status_code}")

    if os.path.getsize(model_path) < 10000:
        raise Exception("Le fichier du modèle semble corrompu ou incomplet.")

    return tf.keras.models.load_model(model_path)

model = download_model()

# === Dictionnaire des classes ===
class_names = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
               'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
               'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish',
               'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
               'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

# === Prétraitement de l’image ===
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# === Upload d’image ===
uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée")

    # Prédiction
    if st.button("🔍 Prédire l’espèce"):
        with st.spinner("Prédiction en cours..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            class_idx = np.argmax(prediction)
            predicted_class = class_names[class_idx]
            confidence = prediction[0][class_idx]

        st.success(f"✅ Espèce prédite : **{predicted_class}** ({confidence*100:.2f}%)")


# === Footer ===
st.markdown("""
    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 0.9em;">
        Réalisé avec 🐟, ☕ et une touche de deep learning par 
        <a href="https://www.linkedin.com/in/hajar-elkhalidi/" target="_blank">Hajar EL KHALIDI</a>, 
        <a href="https://www.linkedin.com/in/aminata-kimbiri-0a9154267/" target="_blank">Aminata KIMBIRI</a> et 
        <a href="https://www.linkedin.com/in/el-idrissi-essaadia-25ab95355/" target="_blank">Essaadia EL IDRISSI</a> — 2025<br>
        <a href="https://github.com/hajar-elkhalidi/deepfish-cnn" target="_blank" style="color:#0366d6; text-decoration: none;">
            🔗 Voir le code sur GitHub
        </a>
    </div>
""", unsafe_allow_html=True)

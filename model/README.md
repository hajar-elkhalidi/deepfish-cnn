# 🐠 Reconnaissance d’espèces de poissons dans des images sous-marines

Ce projet a pour objectif de développer une application d’intelligence artificielle capable de reconnaître automatiquement différentes espèces de poissons à partir d’images sous-marines. Le modèle utilise des techniques de **deep learning** pour classer les poissons parmi 31 espèces différentes.

## 📸 Exemples d'espèces détectées

- Bangus
- Catfish
- Gold Fish
- Tilapia
- Snakehead  
*(Voir la liste complète dans le [dataset](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset) sur Kaggle*

---

## 🛠️ Technologies utilisées

- Python
- Streamlit (interface web)
- TensorFlow / Keras (modèle de classification)
- NumPy, Pandas, etc.

---

## 🚀 Installation

1. **Cloner le dépôt GitHub :**
   ```bash
   git clone https://github.com/hajar-elkhalidi/deepfish-cnn.git
   cd deepfish-cnn
### 2. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
```

#### Activation de l'environnement :

- **Linux / macOS** :
  ```bash
  source venv/bin/activate
  ```
- **Windows** :
  ```bash
  venv\Scripts\activate
  ```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🧪 Utilisation

Lancer l'application Streamlit :

```bash
streamlit run app.py
```

Puis ouvrir l’URL indiquée dans votre terminal (généralement [http://localhost:8501](http://localhost:8501)).

---

## 📁 Structure du projet

```
├── app.py                  # Application Streamlit
├── model/                  # Modèles entraînés
├── images/                 # Images de test
├── requirements.txt        # Dépendances Python
└── README.md               # Fichier de description
```

---

## 👥 Auteurs

Projet réalisé dans le cadre de la Licence d'Excellence en Intelligence Artificielle – 2025, par :

- [Hajar EL KHALIDI](https://www.linkedin.com/in/hajar-elkhalidi/)  
- [Aminata KIMBIRI](https://www.linkedin.com/in/aminata-kimbiri-0a9154267/)  
- [Essaadia EL IDRISSI](https://www.linkedin.com/in/el-idrissi-essaadia-25ab95355/)

---

## 📄 Licence

Ce projet est open-source sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

## 🌐 Démo en ligne

👉 [Voir l'application sur Streamlit Cloud](https://deepfish-cnn.streamlit.app/)
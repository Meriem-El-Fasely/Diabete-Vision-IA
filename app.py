import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration de la page
st.set_page_config(page_title="Diagnostic Diabète IA", layout="wide")

st.title("🏥 Système d'Aide au Diagnostic du Diabète")
st.markdown("""
Cette application utilise le **Machine Learning** pour évaluer le risque de diabète en fonction de mesures médicales.
Elle intègre une **Régression Logistique** pour la prédiction, une **PCA** pour la visualisation et une **Isolation Forest** pour la détection d'anomalies.
""")

# 2. Chargement des modèles sauvegardés
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/model_logistique.pkl')
    pca = joblib.load('models/pca_model.pkl')
    iso_forest = joblib.load('models/iso_forest_model.pkl')
    return scaler, model, pca, iso_forest

try:
    scaler, model, pca, iso_forest = load_models()
except:
    st.error("Erreur : Les fichiers .pkl sont introuvables. Assurez-vous d'avoir exécuté votre notebook d'entraînement.")

# 3. Barre latérale pour la saisie des données
st.sidebar.header("📋 Données du Patient")

def user_input_features():
    pregnancies = st.sidebar.number_input("Grossesses", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Pression Artérielle", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Épaisseur Peau", 0, 100, 20)
    insulin = st.sidebar.slider("Insuline", 0, 900, 80)
    bmi = st.sidebar.slider("IMC (BMI)", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Fonction Pedigree Diabète", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Âge", 1, 100, 30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Traitement et Prédiction
# Standardisation
input_scaled = scaler.transform(input_df.values) 

# Détection d'anomalie (Isolation Forest)
is_anomaly = iso_forest.predict(input_scaled)[0]

# Prédiction (Régression Logistique)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

# 5. Affichage des Résultats
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Résultat de l'Analyse")
    
    # Alerte Anomalie
    if is_anomaly == -1:
        st.warning("⚠️ Attention : Les données saisies semblent atypiques (Anomalie détectée par Isolation Forest).")
    
    # Résultat Diagnostic
    if prediction == 1:
        st.error(f"**Risque Élevé de Diabète**")
    else:
        st.success(f"**Risque Faible de Diabète**")
        
    st.metric("Probabilité de risque", f"{prediction_proba*100:.1f} %")

with col2:
    st.subheader("📍 Visualisation PCA (2D)")
    
    # 1. Chargement et Prétraitement du Dataset pour le fond
    try:
        df = pd.read_csv('data/diabetes.csv')
        
        # Nettoyage (comme dans le notebook)
        cols_to_fix = ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']
        df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        
        X_data = df.drop('Outcome', axis=1).values
        y_data = df['Outcome']
        
        # Transformation
        X_data_scaled = scaler.transform(X_data)
        X_data_pca = pca.transform(X_data_scaled)
        
        # 2. Création du graphique
        fig, ax = plt.subplots()
        
        # Points du dataset (Diabétiques vs Non-Diabétiques)
        # Non-Diabétiques
        ax.scatter(X_data_pca[y_data==0, 0], X_data_pca[y_data==0, 1], 
                   c='blue', label='Non-Diabétique', alpha=0.5, s=20)
        # Diabétiques
        ax.scatter(X_data_pca[y_data==1, 0], X_data_pca[y_data==1, 1], 
                   c='orange', label='Diabétique', alpha=0.5, s=20)
        
        # Point du Patient Actuel
        # Transformation PCA du nouveau point
        input_pca = pca.transform(input_scaled)
        ax.scatter(input_pca[0,0], input_pca[0,1], 
                   c='red', s=200, marker='*', label="Patient Actuel", edgecolors='black')
        
        ax.set_xlabel("Composante Principale 1")
        ax.set_ylabel("Composante Principale 2")
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données pour la visualisation : {e}")
        # Fallback si le dataset n'est pas trouvé
        input_pca = pca.transform(input_scaled)
        fig, ax = plt.subplots()
        ax.scatter(input_pca[0,0], input_pca[0,1], c='red', s=100, label="Patient Actuel")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.info("Note : Cet outil est à but éducatif et ne remplace pas un avis médical professionnel.")

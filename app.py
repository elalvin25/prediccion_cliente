
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Montar Google Drive si es necesario para acceder al modelo cuando se ejecuta fuera de Colab
# from google.colab import drive
# try:
#     drive.mount('/content/drive')
# except: # Para evitar errores si ya está montado o no se ejecuta en Colab
#     pass

# --- Configuración del Modelo y Rutas ---
MODEL_PATH = '/content/drive/MyDrive/cliente/mejor_modelo.pkl'

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Modelo no encontrado en {MODEL_PATH}. "
                 "Asegúrate de que la ruta sea correcta y el archivo exista.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

model = load_model()

# Definir las columnas que el modelo espera (replicado aquí para que el app.py sea autónomo)
x_columns_expected = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PaperlessBilling',
                      'MonthlyCharges', 'InternetService_Fiber optic', 'InternetService_No',
                      'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                      'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                      'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                      'TechSupport_No internet service', 'TechSupport_Yes',
                      'StreamingTV_No internet service', 'StreamingTV_Yes',
                      'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                      'Contract_One year', 'Contract_Two year',
                      'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                      'PaymentMethod_Mailed check']

# CSS personalizado para un diseño atractivo
custom_css = """
<style>
    .main { /* Fondo oscuro */
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .st-emotion-cache-z5fcl4 { /* Fondo del sidebar */
        background-color: #161b22;
        color: #c9d1d9;
    }
    h1, h2, h3, h4, h5, h6 { /* Títulos */
        color: #58a6ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button { /* Botones */
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { /* Botones hover */
        background-color: #2ea043;
        transform: translateY(-2px);
    }
    .stSlider > div > div > div > div { /* Color del slider */
        background-color: #58a6ff;
    }
    .st-emotion-cache-16txt4v { /* Etiqueta del slider */
        color: #c9d1d9;
    }
    .st-emotion-cache-1wv02ee > img { /* Logo en sidebar */
        max-width: 150px;
        margin-bottom: 20px;
        border-radius: 10px;
    }
    /* Alertas */
    .stAlert.success { /* Alerta de no churn */
        background-color: #216e39;
        color: #c9d1d9;
        border-left: 5px solid #2ea043;
        animation: fadeIn 1s ease-in-out;
    }
    .stAlert.error { /* Alerta de churn */
        background-color: #da3633;
        color: #c9d1d9;
        border-left: 5px solid #f85149;
        animation: shake 0.5s;
        animation-iteration-count: 2;
    }
    .stAlert p { /* Texto en alertas */
        font-size: 18px;
        font-weight: bold;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.set_page_config(
    page_title="Sistema de Alerta de Churn - Telco",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Título principal y subtítulo ---
st.markdown("<h1 style='text-align: center; color: #f85149;'>🚨 Sistema de Alerta de Churn - Telco</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #58a6ff;'>Predice el riesgo de abandono de tus clientes</h3>", unsafe_allow_html=True)
st.write("    Esta aplicación utiliza un modelo de Machine Learning (`BaggingClassifier`)     entrenado para predecir si un cliente de telecomunicaciones es propenso a abandonar (churn)     el servicio basándose en sus características y servicios contratados.     Introduce los datos del cliente a continuación y haz clic en 'Analizar Riesgo de Churn'.")

# --- Sidebar con logo y descripción ---
st.sidebar.markdown(
    "<img src='https://www.freeiconspng.com/uploads/telecom-icon-3.png' alt='Logo Telco' style='width: 150px; display: block; margin-left: auto; margin-right: auto;'>",
    unsafe_allow_html=True
)
st.sidebar.header("Acerca del Modelo")
st.sidebar.info(
    "El modelo `BaggingClassifier` fue seleccionado como el de mejor rendimiento     (F1 Score: 0.8487) después de un riguroso proceso de entrenamiento y     optimización de hiperparámetros con técnicas de balanceo de datos (SMOTE).     Las predicciones se basan en un amplio conjunto de características del cliente."
)

# --- Formularios de entrada del usuario ---
with st.form("churn_prediction_form"):
    st.header("Datos del Cliente")

    # 👤 Perfil del Cliente
    st.subheader("👤 Perfil del Cliente")
    senior_citizen = st.radio("¿Es el cliente un Ciudadano Senior?", ("No", "Sí"), horizontal=True)
    partner = st.radio("¿Tiene pareja?", ("No", "Sí"), horizontal=True)
    dependents = st.radio("¿Tiene dependientes?", ("No", "Sí"), horizontal=True)

    # 📱 Servicios Contratados
    st.subheader("📱 Servicios Contratados")
    internet_service = st.selectbox("Servicio de Internet", ("DSL", "Fiber optic", "No"))
    online_security = st.selectbox("Seguridad Online", ("No", "Sí", "No internet service"))
    online_backup = st.selectbox("Backup Online", ("No", "Sí", "No internet service"))
    device_protection = st.selectbox("Protección de Dispositivo", ("No", "Sí", "No internet service"))
    tech_support = st.selectbox("Soporte Técnico", ("No", "Sí", "No internet service"))
    streaming_tv = st.selectbox("Streaming TV", ("No", "Sí", "No internet service"))
    streaming_movies = st.selectbox("Streaming de Películas", ("No", "Sí", "No internet service"))

    # 💳 Información de Cuenta
    st.subheader("💳 Información de Cuenta")
    tenure = st.slider("Antigüedad (meses)", 0, 72, 24) # 0 a 72 meses
    monthly_charges = st.number_input("Cargo Mensual ($", min_value=0.0, max_value=120.0, value=70.0, step=0.1)
    contract = st.selectbox("Tipo de Contrato", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.radio("Facturación Electrónica", ("No", "Sí"), horizontal=True)
    payment_method = st.selectbox("Método de Pago", (
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))

    submitted = st.form_submit_button("🔍 Analizar Riesgo de Churn")

    if submitted:
        # --- Preprocesar la entrada del usuario ---
        input_data = {col: 0 for col in x_columns_expected} # Inicializar con ceros

        # Mapeo de SeniorCitizen, Partner, Dependents, PaperlessBilling (Label Encoding)
        input_data['SeniorCitizen'] = 1 if senior_citizen == 'Sí' else 0
        input_data['Partner'] = 1 if partner == 'Sí' else 0
        input_data['Dependents'] = 1 if dependents == 'Sí' else 0
        input_data['PaperlessBilling'] = 1 if paperless_billing == 'Sí' else 0

        # Mapeo de Tenure y MonthlyCharges (valores numéricos directos)
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly_charges

        # Mapeo de One-Hot Encoding para InternetService
        if internet_service == 'Fiber optic':
            input_data['InternetService_Fiber optic'] = 1
        elif internet_service == 'No':
            input_data['InternetService_No'] = 1

        # Mapeo de One-Hot Encoding para OnlineSecurity, OnlineBackup, etc.
        for service, user_value in [
            ('OnlineSecurity', online_security),
            ('OnlineBackup', online_backup),
            ('DeviceProtection', device_protection),
            ('TechSupport', tech_support),
            ('StreamingTV', streaming_tv),
            ('StreamingMovies', streaming_movies),
        ]:
            if user_value == 'Sí':
                input_data[f'{service}_Yes'] = 1
            elif user_value == 'No internet service':
                input_data[f'{service}_No internet service'] = 1

        # Mapeo de One-Hot Encoding para Contract
        if contract == 'One year':
            input_data['Contract_One year'] = 1
        elif contract == 'Two year':
            input_data['Contract_Two year'] = 1

        # Mapeo de One-Hot Encoding para PaymentMethod
        if payment_method == 'Electronic check':
            input_data['PaymentMethod_Electronic check'] = 1
        elif payment_method == 'Mailed check':
            input_data['PaymentMethod_Mailed check'] = 1
        elif payment_method == 'Credit card (automatic)':
            input_data['PaymentMethod_Credit card (automatic)'] = 1

        # Crear DataFrame para la predicción con el orden correcto de columnas
        df_pred = pd.DataFrame([input_data])[x_columns_expected]

        # --- Realizar Predicción ---
        prediction = model.predict(df_pred)[0]
        prediction_proba = model.predict_proba(df_pred)[0]
        churn_probability = prediction_proba[1] # Probabilidad de Churn (clase 1)

        # --- Mostrar Resultados ---
        st.subheader("Resultado del Análisis de Churn")

        if prediction == 1: # Churn
            st.error("### ¡Alerta! Alto Riesgo de Churn", icon="⚠️")
            st.markdown(
                f"<p style='font-size:18px;'>Este cliente tiene una alta probabilidad de abandonar el servicio. </p>"
                f"<p style='font-size:18px;'>Probabilidad de Churn: <strong>{churn_probability:.2%}</strong></p>",
                unsafe_allow_html=True
            )
            st.warning(
                "**Recomendación:** Considera ofrecerle promociones especiales, mejorar el soporte o                 revisar su plan actual para retenerlo. Un enfoque proactivo es crucial."
            )
        else: # No Churn
            st.success("### Cliente Estable", icon="✅")
            st.markdown(
                f"<p style='font-size:18px;'>Este cliente tiene una baja probabilidad de abandonar el servicio. </p>"
                f"<p style='font-size:18px;'>Probabilidad de No-Churn: <strong>{prediction_proba[0]:.2%}</strong></p>",
                unsafe_allow_html=True
            )
            st.balloons()
            st.info(
                "**Felicitaciones:** El cliente parece estar satisfecho. Sigue monitoreando su                 interacción y busca oportunidades para fortalecer su lealtad."
            )

        st.markdown("### Medidor de Probabilidad de Churn")
        st.progress(churn_probability, text=f"Probabilidad de Churn: {churn_probability:.2%}")


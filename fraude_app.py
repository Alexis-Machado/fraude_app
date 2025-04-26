import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Semilla para reproducibilidad
np.random.seed(42)

# Parámetros de simulación
n_samples = 5000
start_date = datetime.now() - timedelta(days=365)

# Generación de datos
montos = np.random.normal(500, 200, n_samples).clip(0)
fechas = [start_date + timedelta(seconds=np.random.randint(0, 365*24*3600)) for _ in range(n_samples)]
ubic_extranjera = np.random.choice([0,1], n_samples, p=[0.85,0.15])
tipos_comercio = np.random.choice(['fisico','online','atm'], n_samples, p=[0.5,0.4,0.1])
freq_24h = np.random.poisson(2, n_samples)
hist_prom = np.random.normal(800, 300, n_samples).clip(0)
metodo_pago = np.random.choice(['tarjeta','contactless','web'], n_samples, p=[0.6,0.3,0.1])
dispositivo = np.random.choice(['desktop','mobile','tablet'], n_samples, p=[0.4,0.5,0.1])
saldos = np.random.normal(2000, 1000, n_samples).clip(0)
antiguedad = np.random.randint(1,240, n_samples)
autenticacion = np.random.choice([0,1], n_samples, p=[0.2,0.8])

# FRAUDE: 10% de fraudes
fraude = np.random.choice([0,1], n_samples, p=[0.9,0.1])

# Ajustamos los fraudes: Montos muy altos, ubicaciones extranjeras, etc
for i in range(n_samples):
    if fraude[i] == 1:
        montos[i] = np.random.normal(5000, 2000)   # montos anormales
        ubic_extranjera[i] = 1                     # mayormente en el extranjero
        freq_24h[i] = np.random.poisson(10)        # actividad extraña
        autenticacion[i] = 0                       # sin autenticación
        saldo_prob = np.random.rand()
        if saldo_prob > 0.5:
            saldos[i] = np.random.normal(100, 50)  # saldo bajo = sospechoso
        dispositivo[i] = np.random.choice(['desktop', 'mobile', 'tablet'], p=[0.1,0.8,0.1])  # mucho mobile en fraude

# Crear DataFrame
df = pd.DataFrame({
    'monto': montos,
    'fecha_hora': fechas,
    'ubic_extranjera': ubic_extranjera,
    'tipo_comercio': tipos_comercio,
    'frecuencia_24h': freq_24h,
    'historial_promedio': hist_prom,
    'metodo_pago': metodo_pago,
    'dispositivo': dispositivo,
    'saldo_cuenta': saldos,
    'antiguedad_meses': antiguedad,
    'autenticacion': autenticacion,
    'fraude': fraude
})

# Ingeniería de variables temporales
df['hora'] = df['fecha_hora'].dt.hour
df['dia_semana'] = df['fecha_hora'].dt.dayofweek
df['mes'] = df['fecha_hora'].dt.month

# Definición de X e y
X = df.drop(columns=['fecha_hora','fraude'])
y = df['fraude']

# Preprocesamiento
numeric_feats = ['monto','frecuencia_24h','historial_promedio','saldo_cuenta','antiguedad_meses','hora','dia_semana','mes']
cat_feats = ['ubic_extranjera','tipo_comercio','metodo_pago','dispositivo','autenticacion']

numeric_trans = StandardScaler()
categorical_trans = OneHotEncoder(drop='if_binary', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', numeric_trans, numeric_feats),
    ('cat', categorical_trans, cat_feats)
])

X_proc = preprocessor.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

# Definición del modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
with st.spinner('Entrenando modelo...'):
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

# Evaluación
loss, acc = model.evaluate(X_test, y_test, verbose=0)
st.success(f"✅ Modelo entrenado - Precisión en test: {acc:.4f}")

st.title("💳 Sistema de Detección de Fraudes")

st.header("Datos de la transacción")

# Entradas del usuario
monto = st.number_input('Monto de la transacción', min_value=0.0, value=500.0)
ubic_extranjera = st.selectbox('¿Ubicación extranjera?', [0, 1])
tipo_comercio = st.selectbox('Tipo de comercio', ['fisico', 'online', 'atm'])
frecuencia_24h = st.slider('Frecuencia de transacciones últimas 24h', 0, 50, 2)
historial_promedio = st.number_input('Historial promedio de gastos', min_value=0.0, value=800.0)
metodo_pago = st.selectbox('Método de pago', ['tarjeta', 'contactless', 'web'])
dispositivo = st.selectbox('Dispositivo usado', ['desktop', 'mobile', 'tablet'])
saldo_cuenta = st.number_input('Saldo en la cuenta', min_value=0.0, value=2000.0)
antiguedad_meses = st.slider('Antigüedad de la cuenta (meses)', 1, 240, 24)
autenticacion = st.selectbox('¿Autenticación realizada?', [0, 1])

if st.button('🔍 Evaluar transacción'):
    ejemplo = pd.DataFrame([{
        'monto': monto,
        'ubic_extranjera': ubic_extranjera,
        'tipo_comercio': tipo_comercio,
        'frecuencia_24h': frecuencia_24h,
        'historial_promedio': historial_promedio,
        'metodo_pago': metodo_pago,
        'dispositivo': dispositivo,
        'saldo_cuenta': saldo_cuenta,
        'antiguedad_meses': antiguedad_meses,
        'autenticacion': autenticacion,
        'hora': datetime.now().hour,
        'dia_semana': datetime.now().weekday(),
        'mes': datetime.now().month
    }])
    
    X_ej_proc = preprocessor.transform(ejemplo)
    prob = model.predict(X_ej_proc)[0,0]
    
    st.metric(label="Probabilidad de Fraude", value=f"{prob:.2%}")
    
    if prob > 0.5:
        st.error("🚨 ¡Alerta de posible fraude!")
    else:
        st.success("✅ Transacción segura.")

<p align="center">
  <img src="https://www.iudigital.edu.co/images/11.-IU-DIGITAL.png" alt="IU Digital" width="350">
</p>

<p align="center">
  🚀 Aplicación desplegada:  
  <a href="https://alexis-machado-fraude-app-fraude-app-xxpz1c.streamlit.app/" target="_blank">
    https://alexis-machado-fraude-app-fraude-app-xxpz1c.streamlit.app/
  </a>
</p>

# 💳 Simulación Sistema de Detección de Fraudes con Deep Learning

¡Bienvenido al proyecto Simulación de detección de fraudes financieros usando Deep Learning! 🎉

Este repositorio contiene una **aplicación web** desarrollada en **Streamlit**, junto con la **simulación** de datos y un **modelo de red neuronal** construido con **TensorFlow/Keras**. El objetivo es detectar transacciones fraudulentas en tiempo real. ⚡️

---

## 📋 Contenido

- 🧩 [Descripción](#-descripción)
- 🛠️ [Tecnologías y Dependencias](#️-tecnologías-y-dependencias)
- 🚀 [Instalación](#-instalación)
- ▶️ [Uso](#️-uso)
- 🔍 [Cómo Funciona](#-cómo-funciona)
- 📂 [Estructura del Proyecto](#-estructura-del-proyecto)
- 📈 [Simulación de Datos](#-simulación-de-datos)
- 🧠 [Modelo de Deep Learning](#-modelo-de-deep-learning)

---

## 📖 Descripción

Este proyecto simula transacciones financieras y entrena una red neuronal para clasificar cada operación como legítima o fraudulenta. La aplicación permite al usuario ingresar parámetros de una transacción y obtener una probabilidad de fraude al instante.

---

## 🛠️ Tecnologías y Dependencias

- Python 3.8+
- Streamlit
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib (opcional para visualizaciones internas)

Instala todo con:
```bash
pip install -r requirements.txt
```

---

## 🚀 Instalación

1. **Clona el repositorio**:
   ```bash
   https://github.com/Alexis-Machado/fraude_app.git
   ```

2. **Crea un entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate    # Windows
   ```

   **NOTA:** Si prefieres, también puedes instalar las dependencias directamente en tu entorno local sin usar un entorno virtual.

3. **Instala dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Uso

Para ejecutar la aplicación Streamlit:
```bash
streamlit run fraude_app.py
```

1. Abre tu navegador en `http://localhost:8501`
2. Completa los campos de la transacción:
   - Monto
   - Ubicación extranjera
   - Tipo de comercio
   - Frecuencia en últimas 24h
   - Historial promedio
   - Dispositivo, método de pago, etc.
3. Haz click en **Predecir** y obtén la probabilidad de fraude.

---

## 🔍 Cómo Funciona

1. **Generación de datos**: Se crea un DataFrame con transacciones simuladas (legítimas vs. fraudulentas).
2. **Preprocesamiento**: Normalización de variables numéricas y codificación de categorías.
3. **Entrenamiento**: Red neuronal con dos capas ocultas (ReLU) y capa de salida Sigmoide.
4. **Evaluación**: Cálculo de precisión sobre conjunto de prueba.
5. **Predicción**: Streamlit recibe entrada del usuario y muestra probabilidad de fraude.

---

## 📂 Estructura del Proyecto

```bash
fraude_app/
├── fraude_app.py        # Aplicación principal en Streamlit
├── requirements.txt     # Dependencias
├── README.md            # Documentación 

```

---

## 📈 Simulación de Datos

- Usamos `numpy` y `pandas` para generar 5,000 transacciones.
- Variables: monto, fecha_hora, ubicación, tipo de comercio, frecuencia, historial, dispositivo, autenticación, etc.
- Fraudes generados con probabilidad del 10% y características anómalas.

---

## 🧠 Modelo de Deep Learning

- Estructura:
  - Input: 12 características preprocesadas
  - Capa oculta 1: 64 neuronas, ReLU
  - Capa oculta 2: 32 neuronas, ReLU
  - Salida: 1 neurona, Sigmoide
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Métrica: Accuracy

Entrena durante 20 épocas con batch size de 64.

---

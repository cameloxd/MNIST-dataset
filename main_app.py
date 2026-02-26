import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Configuración inicial
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

st.title("🔢 Clasificador de Dígitos MNIST (Demo Urgente)")
st.markdown("Esta app entrena un modelo en tiempo real para clasificar números escritos a mano (0-9).")

# --- CARGA DE DATOS ---
digits = load_digits()
X = digits.data
y = digits.target

# --- SIDEBAR: Configuración ---
st.sidebar.header("🛠️ Configuración")
model_type = st.sidebar.selectbox(
    "Selecciona el Modelo:",
    ("Random Forest", "SVM (Support Vector Machine)", "KNN (K-Nearest Neighbors)")
)

test_size = st.sidebar.slider("Porcentaje de prueba", 0.1, 0.4, 0.2)

# --- ENTRENAMIENTO ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)
elif model_type == "SVM (Support Vector Machine)":
    model = SVC(probability=True)
else:
    model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- MÉTRICAS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📊 Desempeño: {model_type}")
    st.metric("Precisión del Modelo", f"{acc*100:.2f}%")
    
    # Matriz de Confusión
    st.write("**Matriz de Confusión:**")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    st.pyplot(fig)

# --- VALIDACIÓN INDIVIDUAL (Requisito 3) ---
with col2:
    st.subheader("🔍 Validación de Imágenes")
    st.write("Selecciona un índice del set de prueba para validar:")
    
    idx = st.number_input("Índice de imagen (0 a 300):", 0, 300, 10)
    
    # Mostrar la imagen
    sample_img = X_test[idx].reshape(8, 8)
    real_label = y_test.iloc[idx] if isinstance(y_test, pd.Series) else y_test[idx]
    prediction = model.predict([X_test[idx]])[0]
    
    fig_img, ax_img = plt.subplots()
    ax_img.imshow(sample_img, cmap='gray_r')
    ax_img.axis('off')
    st.pyplot(fig_img)
    
    if prediction == real_label:
        st.success(f"✅ Predicción: {prediction} | Real: {real_label}")
    else:
        st.error(f"❌ Predicción: {prediction} | Real: {real_label}")

# --- REPORTE DETALLADO ---
st.divider()
with st.expander("Ver Reporte de Clasificación"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
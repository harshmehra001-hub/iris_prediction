import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Iris AI Predictor",
    page_icon="🌸",
    layout="wide"
)


# ---------------- TITLE ----------------
st.title("🌸 Iris AI Predictor")
st.subheader("Deep Learning based Iris Flower Classification")

st.write(
    "This application uses a TensorFlow/Keras Artificial Neural Network "
    "to classify Iris flowers into three different species."
)


# ---------------- MODEL INFORMATION ----------------
st.header("🤖 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Input Features", "4")

with col2:
    st.metric("Classes", "3")

with col3:
    st.metric("Model", "ANN")

with col4:
    st.metric("Type", "AI Prediction")


# ---------------- LOAD MODEL & SCALER ----------------
@st.cache_resource
def load_resources():

    model_path = "iris_mode.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' project folder me nahi mili."
        )

    # TensorFlow ke through model load hoga
    model = tf.keras.models.load_model(model_path)

    scaler_path = "scaler.pkl"

    if os.path.exists(scaler_path):

        scaler = joblib.load(scaler_path)

    else:

        excel_path = "iris.xlsx"

        if not os.path.exists(excel_path):
            raise FileNotFoundError(
                "scaler.pkl aur iris.xlsx dono nahi mile."
            )

        df = pd.read_excel(excel_path)

        X = df.drop(columns=["class"])

        scaler = StandardScaler()
        scaler.fit(X)

    return model, scaler


# ---------------- LOAD RESOURCES ----------------
try:

    model, scaler = load_resources()

except Exception as e:

    st.error("❌ Model load nahi ho raha.")
    st.code(str(e))

    st.info(
        """
        Check karo:

        1. iris_mode.keras file project folder me hai
        2. iris.xlsx file project folder me hai
        3. TensorFlow installed hai
        """
    )

    st.stop()


# ---------------- INPUT SECTION ----------------
st.header("🌱 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )


with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Iris Species", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepallength",
            "sepalwidth",
            "petallength",
            "petalwidth"
        ]
    )

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        np.max(prediction[0]) * 100
    )


    # ---------------- CLASS NAMES ----------------
    class_names = [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ]

    predicted_class = class_names[predicted_index]


    # ---------------- RESULT ----------------
    st.success(
        f"🌸 Predicted Species: **{predicted_class}**"
    )

    st.info(
        f"🎯 Confidence: **{confidence:.2f}%**"
    )


    # ---------------- PROBABILITY ----------------
    st.subheader("📊 Prediction Probabilities")

    probability_data = pd.DataFrame(
        {
            "Species": class_names,
            "Probability": prediction[0] * 100
        }
    )

    st.bar_chart(
        probability_data.set_index("Species")
    )
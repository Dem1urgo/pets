import streamlit as st
import pandas as pd
import joblib
# importamos la libreria

st.title("Pets")
# titulo de la pagina

st.write("Ejemplo de predicción de mascota.")

# Input bar 1
height = st.number_input("Introduce la altura:")

# Input bar 2
weight = st.number_input("Introduce el peso:")

# dropdwn input
eyes = st.selectbox('Escoge el color de los ojos', ('Blue', 'Brown'))

st.write('Has escogido:', eyes)



if st.button('Submit'):

    # unplickle classifier
    pet_model = joblib.load("petmodel.pkl")
    X = pd.DataFrame([[height, weight, eyes]], columns = ["Height", "Weight", "Eye"])
    X = X.replace(["Blue", "Brown"], [1, 0])
    
    # get prediction
    prediction = pet_model.predict(X)[0]

    # output prediction
    st.text(f"This instance is a {prediction}")
import streamlit as st
import joblib
from keras.saving import load_model
import numpy as np
import PIL.Image
import PIL.ImageOps
import io

# Load model and scaler for CBC prediction
cbc_model = load_model('CBC_V2_Model.keras')
scaler_cbc = joblib.load('std_scaler_v2model.bin')

# Anemia class names for CBC prediction
class_names_cbc = {
    0: 'Normal',
    1: 'HGB Anemia',
    2: 'Iron Anemia',
    3: 'Folate Anemia',
    4: 'B12 Anemia'
}

# Function for CBC prediction
def model_v2_predict(inputs):
    inputs_scaled = scaler_cbc.transform(np.array(inputs).reshape(1, -1))
    pred = cbc_model.predict(inputs_scaled)
    label = pred.argmax(axis=1)[0]
    return class_names_cbc[label]

# Load kidney model
kideny_model = load_model('kideny_model.keras')
class_names_kideny = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Streamlit Interface
st.set_page_config(page_title="Healthcare Models", page_icon="🏥")
st.title("Healthcare Model Predictions")

# Sidebar for navigation
st.sidebar.title("Choose Prediction Type")
prediction_type = st.sidebar.radio("Choose model:", ["CBC Prediction", "Kidney Prediction"])

# CBC Prediction form
if prediction_type == "CBC Prediction":
    st.subheader("Enter CBC Test Data")

    gender = st.selectbox("Gender", [0, 1])  # 0 = Male, 1 = Female
    wbc = st.number_input("WBC (White Blood Cells)", min_value=0.0, step=0.1)
    ne = st.number_input("NE (Neutrophils)", min_value=0.0, step=0.1)
    ly = st.number_input("LY (Lymphocytes)", min_value=0.0, step=0.1)
    mo = st.number_input("MO (Monocytes)", min_value=0.0, step=0.1)
    eo = st.number_input("EO (Eosinophils)", min_value=0.0, step=0.1)
    ba = st.number_input("BA (Basophils)", min_value=0.0, step=0.1)
    rbc = st.number_input("RBC (Red Blood Cells)", min_value=0.0, step=0.1)
    hgb = st.number_input("HGB (Hemoglobin)", min_value=0.0, step=0.1)
    hct = st.number_input("HCT (Hematocrit)", min_value=0.0, step=0.1)
    mcv = st.number_input("MCV (Mean Corpuscular Volume)", min_value=0.0, step=0.1)
    mch = st.number_input("MCH (Mean Corpuscular Hemoglobin)", min_value=0.0, step=0.1)
    mchc = st.number_input("MCHC (Mean Corpuscular Hemoglobin Concentration)", min_value=0.0, step=0.1)
    rdw = st.number_input("RDW (Red Cell Distribution Width)", min_value=0.0, step=0.1)
    plt = st.number_input("PLT (Platelets)", min_value=0.0, step=0.1)
    mpv = st.number_input("MPV (Mean Platelet Volume)", min_value=0.0, step=0.1)
    pct = st.number_input("PCT (Platelet Crit)", min_value=0.0, step=0.1)
    pdw = st.number_input("PDW (Platelet Distribution Width)", min_value=0.0, step=0.1)
    sd = st.number_input("SD (Standard Deviation)", min_value=0.0, step=0.1)
    sdtsd = st.number_input("SDTSD (Standard Deviation Test)", min_value=0.0, step=0.1)
    tsd = st.number_input("TSD (Test SD)", min_value=0.0, step=0.1)
    ferr = st.number_input("Ferritin", min_value=0.0, step=0.1)
    folate = st.number_input("Folate", min_value=0.0, step=0.1)
    b12 = st.number_input("B12", min_value=0.0, step=0.1)

    if st.button("Predict CBC"):
        # Make prediction
        inputs = [
            int(gender), float(wbc), float(ne), float(ly), float(mo), float(eo), float(ba),
            float(rbc), float(hgb), float(hct), float(mcv), float(mch), float(mchc), float(rdw),
            float(plt), float(mpv), float(pct), float(pdw), float(sd), float(sdtsd), float(tsd),
            float(ferr), float(folate), float(b12)
        ]
        result = model_v2_predict(inputs)
        st.write(f"Prediction: {result}")

# Kidney Prediction form
if prediction_type == "Kidney Prediction":
    st.subheader("Upload Kidney Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        contents = uploaded_file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))
        pil_image = pil_image.resize((128, 128), PIL.Image.Resampling.LANCZOS)
        img_arr = np.array(pil_image)
        img_arr_norm = img_arr / 255.0
        img_arr_norm = np.expand_dims(img_arr_norm, axis=0)

        # Make prediction
        prediction = kideny_model.predict(img_arr_norm)
        predicted_class = np.argmax(prediction, axis=1)

        st.write(f"Prediction: {class_names_kideny[predicted_class[0]]}")
        st.image(pil_image, caption='Uploaded Image', use_container_width=True)
        

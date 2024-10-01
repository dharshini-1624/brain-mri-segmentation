import streamlit as st
import requests
from PIL import Image
import io


st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    with st.spinner("Segmenting..."):
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            files={"file": uploaded_file}
        )

   
    if response.status_code == 200:
        result_image = Image.open(io.BytesIO(response.json()["segmentation"]))
        st.image(result_image, caption='Segmentation Result', use_column_width=True)
    else:
        st.error("Error: " + response.json()["error"])

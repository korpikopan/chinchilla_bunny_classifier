import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from label_image import bunny_or_chinch

# settings
st.set_page_config(page_title="Chinchilla Bunny Classifier v1", page_icon=":tada:", layout="wide")




with st.sidebar:
    uploaded_img = st.file_uploader("Upload a picture", type=["png", "jpg"])


st.header("Chinchilla Bunny Classifier")
if uploaded_img:
    st.success("image successfully uploaded")
    st.write("---")
    st.image(uploaded_img, width=240)
    st.write("---")
    button = st.button("Classifier")
    if button:
        result = bunny_or_chinch(uploaded_img)
        st.write(result)

import streamlit as st
from streamlit_option_menu import option_menu

from label_image import bunny_or_chinch
from PIL import Image

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
        img = uploaded_img.read()
        results = bunny_or_chinch(img)
        _max = [res[1] for res in results]
        st.write("##")
        st.success("Process done!")
        for res in results:
            lapin_or_chinchilla, score = res
            st.write('%-20s : %.5f' % (lapin_or_chinchilla, score))
        
        #animal, score = results[_max.index(max(_max))]
        #st.write("c'est un %-20s à %.5f" % (animal, score))

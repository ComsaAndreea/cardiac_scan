import streamlit as st
from PIL import Image

st.title("Cardiac Scan AI")
st.write("Upload a cardiac MRI/CT image for analysis")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        st.write("AI model would analyze the image here")
        st.success("Analysis complete (placeholder)")
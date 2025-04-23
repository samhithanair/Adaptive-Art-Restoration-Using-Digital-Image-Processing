import streamlit as st
import numpy as np
import cv2
from PIL import Image
from restore import adaptive_restoration

st.title("Image Restoration Tool")
st.text("Samhitha Nair 21BAI1183 & Nigel Joe Tensing 21BAI1267")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image_np, caption="Original Image", use_container_width=True)

    if st.button("Run Adaptive Restoration"):
        restored_cv2, log = adaptive_restoration(image_cv2)
        restored_rgb = cv2.cvtColor(restored_cv2, cv2.COLOR_BGR2RGB)
        st.image(restored_rgb, caption="Restored Image", use_container_width=True)

        st.markdown("### Processing Log")
        for step in log:
            st.write(f"✅ {step}")

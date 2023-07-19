import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


st.write("### Quick app for visualization of SVD process")
st.write("")

img = plt.imread('panda.JPG')


st.sidebar.header('Input Parameters')
st.write("")
uploaded_file = st.file_uploader("Загрузите вашу фотографию", type=["jpg", "jpeg", "png"])

st.sidebar.write(f'Choose K for SVD, any number from 1 to 100')
st.write("")
st.write("")
st.write("")
k = st.sidebar.slider('Parameter for SVD', 0, 100, 2)


if uploaded_file is not None:
    # Отображение изображения
    img = Image.open(uploaded_file)
    img = img.resize((500, 500))
else:
    img = plt.imread('panda.JPG')

# img = np.array(img)[:, :, 0]
img = np.array(img, dtype=np.float32) / 255  # Преобразование и нормализация изображения

U, sing_vals, V = np.linalg.svd(img)
sigma = np.zeros(shape=(U.shape[0], V.shape[0]))
np.fill_diagonal(sigma, sing_vals)

top_k = k

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
# trunc_V = V[:top_k, :]
trunc_V = V[:, :top_k]  # Исправление размерности trunc_V

# trunc_img = trunc_U@trunc_sigma@trunc_V
trunc_img = trunc_U @ trunc_sigma @ trunc_V.T 

st.write('###### Guese WHO?')
st.write("")

# Отображение изображения в Matplotlib
st.image(trunc_img, clamp=True)
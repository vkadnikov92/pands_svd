import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.write("### Quick app for visualization of SVD process")
st.write("")

img = plt.imread('panda.JPG')


st.sidebar.header('Input Parameters')
st.sidebar.write(f'Choose K for SVD, any number from 1 to 100')
st.write("")
st.write("")
st.write("")
k = st.sidebar.slider('Parameter for SVD', 0, 100, 2)
# k = st.sidebar.slider('k parameter for SVD', min_value=1, max_value=100)

img = plt.imread('panda.JPG')
img = img[:, :, 0]
img = img/255
U, sing_vals, V = np.linalg.svd(img)
sigma = np.zeros(shape=(U.shape[0], V.shape[0]))
np.fill_diagonal(sigma, sing_vals)

top_k = k

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

trunc_img = trunc_U@trunc_sigma@trunc_V

st.write('###### Guese WHO?')
st.write("")

# Отображение изображения в Matplotlib
st.image(trunc_img, clamp=True)
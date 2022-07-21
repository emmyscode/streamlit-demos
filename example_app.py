import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout='wide')

st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: <https://emmyscode-streamlit-example-app-ojc2pw.streamlitapp.com>
    \n
    GitHub Respository: <https://github.com/emmyscode/streamlit>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Inspirit AI: <https://www.inspiritai.com>
    [GitHub](https://github.com/emmyscode) | [LinkedIn](https://www.linkedin.com/company/inspirit-ai/)
    """
)
st.title("Sketch Recognition")
st.header("Classifying Hand-Drawn Doodles with a Neural Network")
st.markdown(
    """
    Welcome to this example app created using [streamlit](https://streamlit.io)! Here, we will perform some simple data visualizations as well as deploy a simple machine learning model.
    """
)

def get_canvas_size():
    width = 500
    height = 500
    return width, height

def get_app_response(classification, probability):
    if classification == "Apple":
        st.write("Apple")
    elif classification == "Banana":
        st.write("Banana")
    elif classification == "Carrot":
        st.write("Carrot")
    else:
        st.write("Oops")

class_names = np.genfromtxt("labels.txt", dtype="str", delimiter='\n')
for i in range(len(class_names)):
  class_names[i] = ' '.join(class_names[i].split(' ')[1::])
np.set_printoptions(suppress=True)

model = tf.keras.models.load_model('keras_model.h5', compile = False)

def get_image_classification(image_data):
    img = image_data
    img32 = np.float32(img)
    bgr = cv2.cvtColor(img32, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(rgb,(224,224)).astype('float32')
    rgb = np.reshape(rgb,(1,224,224,3))
    rgb = rgb / 255

    predictions = model.predict(rgb)[0]
    classification = class_names[np.argmax(predictions)]
    return classification, predictions

width, height = get_canvas_size()

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color= bg_color,
    # background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    width = width,
    height=height,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.json_data:
  if canvas_result.json_data["objects"]:
    if canvas_result.image_data is not None:
      img = canvas_result.image_data
      classification, probability = get_image_classification(img)
      st.subheader("Classification: " + classification)
      get_app_response(classification, probability)
  else:
    st.subheader("Draw Something!")


import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import base64
from PIL import Image, ImageDraw
import numpy as np
import cv2
from io import StringIO
import json
import requests
import streamlit as st
from utils import decode_img, encode_image
from utils import get_ellipse_coords
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: red;'>OBJECT DETECTION WITH TRANSFORMER</h1>",
            unsafe_allow_html=True)


with st.sidebar:
    threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5)
    add_radio = st.radio(label="Choose model",
                         options=("res-50_ddetr",
                                  "mb-v3L_ddetr",
                                  "effi-v2S_ddetr",
                                  "swin-T_ddetr",
                                  "all models"))


if 'button' not in st.session_state:
    st.session_state["button"] = False

if "points" not in st.session_state:
    st.session_state["points"] = []


def reset_point():
    st.session_state["points"] = []


def recognize(url="http://192.168.1.4:4500/app/detect", body={}):
    step_by_step = False
    if add_radio == "res-50_ddetr":
        st.markdown("""<p style="font-size:20px">resnet_50_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "mb-v3L_ddetr":
        st.markdown("""<p style="font-size:20px">mobilenet_v3_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "effi-v2S_ddetr":
        st.markdown("""<p style="font-size:20px">efficientnet_v2s_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "swin-T_ddetr":
        st.markdown("""<p style="font-size:20px">swin_transformer_t_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "all models":
        st.markdown("""<p style="font-size:20px">all models</p>""",
                    unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose image file.", type=[
        'png', 'jpg', 'jpeg', 'PNG', 'JPEG'], accept_multiple_files=False, on_change=reset_point)

    if add_radio == "all models":
        step_by_step = st.checkbox("Step by step")

    st.markdown(
        """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    ready_to_recognize = True
    image_base64 = []
    if uploaded_file != None:

        image_pil = Image.open(uploaded_file)
        image = np.array(image_pil)
        if not step_by_step:
            with col1:
                st.markdown("**Input:**")
                st.image(image)
        else:
            ready_to_recognize = False
            with col1:
                st.markdown("**Input:**")
                draw = ImageDraw.Draw(image_pil)
                value = streamlit_image_coordinates(
                    image_pil, key=None)

                if value is not None:
                    point = value["x"], value["y"]
                    st.session_state["points"].append(point)

                # Draw an ellipse at each coordinate in points
                for point in st.session_state["points"]:
                    coords = get_ellipse_coords(point)
                    draw.ellipse(coords, fill="red")
            with col2:
                st.markdown(
                    "**Click on input image to choose attention point:**")
                image_placeholder = st.empty()
                image_placeholder.image(image_pil)

                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    finish_button = st.button("Finish select point")
                with col2_2:
                    clear_button = st.button("Clear attention points")

                if finish_button:
                    ready_to_recognize = True

                if clear_button:
                    st.session_state["points"] = []
                    image_placeholder.image(image)
                st.markdown(st.session_state["points"])

        image_base64 = encode_image(image)

        if ready_to_recognize:
            body = {
                "document": image_base64,
                "threshold": threshold,
                "model_name": add_radio,
                "step_by_step": step_by_step,
                "points": st.session_state["points"]
            }
            body = json.dumps(body)
            data = requests.post(url, data=body)
            uploaded_file = None
            return data.text
    return None


if add_radio in ["res-50_ddetr", "mb-v3L_ddetr", "effi-v2S_ddetr", "swin-T_ddetr", "all models"]:
    data = recognize()
    if data is not None:
        data = json.loads(data)

        if data["step_by_step"]:

            st.markdown(
                """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
            st.markdown("**Encoder:**")

            encoder = data["attention"]["encoder"]
            for k, v in encoder.items():
                if k == "swin-T_ddetr":
                    continue

                st.text(k)
                st.image(decode_img(v))

            st.markdown(
                """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
            st.markdown("**Decoder:**")

            decoder = data["attention"]["decoder"]
            for k, v in decoder.items():
                if k == "swin-T_ddetr":
                    continue

                st.text(k)
                st.image(decode_img(v))

        st.markdown(
            """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
        st.markdown("**Output:**")
        if len(data['result']) > 0:
            result = decode_img(data['result'])
            st.image(result)
        st.code("message: " + data['message'])

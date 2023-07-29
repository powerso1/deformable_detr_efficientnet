
import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import base64
from PIL import Image
import numpy as np
import cv2
from io import StringIO
import json
import requests
import streamlit as st
from utils import decode_img


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
        step_by_step = st.checkbox("Step by step")

    uploaded_file = st.file_uploader("Choose image file.", type=[
        'png', 'jpg', 'jpeg', 'PNG', 'JPEG'], accept_multiple_files=False)
    st.markdown(
        """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
    image_base64 = []
    if uploaded_file != None:
        st.markdown("**Input:**")
        image = Image.open(uploaded_file)

        image = np.array(image)

        # h, w, _ = image.shape

        # ratio = 800 / w
        # image = cv2.resize(image, (800, int(h * ratio)))
        st.image(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, im_arr = cv2.imencode('.jpg', image)
        im_bytes = im_arr.tobytes()
        b64_string = base64.b64encode(im_bytes)
        image_base64 = b64_string.decode("utf-8")

        body = {
            "document": image_base64,
            "threshold": threshold,
            "model_name": add_radio,
            "step_by_step": step_by_step
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

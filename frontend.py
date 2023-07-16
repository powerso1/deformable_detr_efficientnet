
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
from src.utils import decode_img


st.markdown("<h1 style='text-align: center; color: red;'>YOLO V7 FRUIT DETECTION</h1>",
            unsafe_allow_html=True)

with st.sidebar:
    add_radio = st.radio(label="Tab",
                         options=("Thông tin nhóm",
                                  "yolov7 original",
                                  "yolov7 fruit"))


def recognize(url="http://192.168.1.4:4000/yolov7/fruit", body={}):
    if add_radio == "yolov7 original":
        st.markdown("""<p style="font-size:20px">Detected class: 80 class from COCO dataset</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "yolov7 fruit":
        st.markdown("""<p style="font-size:20px">Detected class: 6 class include pineapple, cherry, plum, mango, watermelon, tomatoes</p>""",
                    unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose image file. Image with aspect ratio 1:1 is recomended.", type=[
        'png', 'jpg', 'jpeg', 'PNG', 'JPEG'], accept_multiple_files=False)
    st.markdown(
        """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
    image_base64 = []
    if uploaded_file != None:
        st.markdown("**Input:**")
        image = Image.open(uploaded_file)

        image = np.array(image)

        h, w, _ = image.shape

        ratio = 800 / w
        image = cv2.resize(image, (800, int(h * ratio)))
        st.image(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, im_arr = cv2.imencode('.jpg', image)
        im_bytes = im_arr.tobytes()
        b64_string = base64.b64encode(im_bytes)
        image_base64 = b64_string.decode("utf-8")

        if add_radio == "yolov7 original":
            model_type = "original"
        elif add_radio == "yolov7 fruit":
            model_type = "fruit"

        body = {
            "model_type": model_type,
            "document": image_base64
        }
        body = json.dumps(body)
        data = requests.post(url, data=body)
        uploaded_file = None
        return data.text
    return None


if add_radio == "Thông tin nhóm":
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        img = cv2.imread("../profile/duc.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (480, 640))
        st.image(img)
        st.markdown(
            """<p style="font-size:30px;text-align:center">Hoàng Minh Đức \n 19127121</p>""", unsafe_allow_html=True)
    with col2:
        img = cv2.imread("../profile/hien.png")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (480, 640))
        st.image(img)
        st.markdown(
            """<p style="font-size:30px;text-align:center">Nguyễn Hữu Hiển \n 19127394</p>""", unsafe_allow_html=True)
    with col3:
        img = cv2.imread("../profile/thanh.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (480, 640))

        st.image(img)
        st.markdown(
            """<p style="font-size:30px;text-align:center">Tống Tất Thành \n 19127549</p>""", unsafe_allow_html=True)

if add_radio in ["yolov7 original", "yolov7 fruit"]:
    data = recognize()
    if data is not None:
        data = json.loads(data)
        st.markdown(
            """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
        st.markdown("**Output:**")
        st.code("message: " + data['message'])
        if len(data['result']) > 0:
            result = decode_img(data['result'])
            st.image(result)

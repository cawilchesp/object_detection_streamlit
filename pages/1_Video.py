from ultralytics import YOLO, RTDETR
import supervision as sv
import streamlit as st

import cv2
import yaml
import torch
import time
from pathlib import Path
import itertools

from imutils.video import FileVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink

import config

# For debugging
from icecream import ic

st.title('Video Detection')


# SIDEBAR
st.sidebar.title('Model')

source = st.sidebar.file_uploader(
    label="Choose a video...",
    type=("mp4", "avi")
)

model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

if model_type:
    if 'v8' in model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, 'yolov8', str(model_type))
    elif 'v9' in model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, 'yolov9', str(model_type))
else:
    st.error('Select Model in Sidebar')

confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))

# VIDEO PAGE

if source:
    st.video(source)

    if st.button("Execution"):
        print('play')
    else: print('stop')

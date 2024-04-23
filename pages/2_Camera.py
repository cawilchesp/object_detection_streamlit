from ultralytics import YOLO, RTDETR
import supervision as sv
import streamlit as st

import cv2
import yaml
import torch
import time
from pathlib import Path
import itertools
import tempfile

from imutils.video import FileVideoStream
from imutils.video import FPS

import config
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink


# For debugging
from icecream import ic

st.set_page_config(layout="wide")

st.title('Webcam Detection')

# SIDEBAR
st.sidebar.header("Model")

model_type = st.sidebar.selectbox(
    "Select Model",
    config.MODEL_LIST
)

confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, config.DETECTION_MODEL_DICT[model_type])
else:
    st.error('Select Model in Sidebar')

# load pretrained DL model
try:
    model = config.load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
col_image, col_info_1, col_info_2 = st.columns([5, 1, 1])
with col_image:
    st_frame = st.empty()
with col_info_1:
    st.markdown('**Width**')
    st.markdown('**Height**')
    st.markdown('**Frame Rate**')
    st.markdown('**Frame**')
with col_info_2:
    width_text = st.markdown('0 px')
    height_text = st.markdown('0 px')
    fps_text = st.markdown('0 FPS')
    frame_text = st.markdown('0')

col_play, col_stop, col3 = st.columns([1, 1, 5])
with col_play:
    play_button = st.button(label="Play Webcam")
with col_stop:
    stop_button = st.button(label="Stop Webcam")

play_flag = False
if play_button: play_flag = True
if stop_button: play_flag = False
with st.spinner("Running..."):
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width_text.write(f"{width} px")
    height_text.write(f"{height} px")
    fps_text.write(f"{fps:.2f} FPS")

    # Annotators
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(width, height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(width, height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)

    frame_number = 0
    while play_flag:
        success, image = cap.read()
        if not success:break
        frame_text.write(str(frame_number))
        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720 * (9 / 16))))

        results = model.track(
            source=image,
            persist=True,
            imgsz=640,
            conf=confidence,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            retina_masks=True,
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms()

        # Draw labels
        object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
        annotated_image = label_annotator.annotate(
            scene=image,
            detections=detections,
            labels=object_labels )
        
        # Draw boxes
        annotated_image = bounding_box_annotator.annotate(
            scene=annotated_image,
            detections=detections )
        
        # Draw tracks
        if detections.tracker_id is not None:
            annotated_image = trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )

        st_frame.image(
            annotated_image,
            caption='Detected Video',
            channels="BGR",
            use_column_width=True
        )
        frame_number += 1
    cap.release()
    
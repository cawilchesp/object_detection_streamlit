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
    
    model = config.load_model(model_path)

else:
    st.error('Select Model in Sidebar')

confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))




# VIDEO PAGE

if source:
    cap = cv2.VideoCapture()
    tffile = tempfile.NamedTemporaryFile()
    tffile.write(source.read())
    cap.open(tffile.name)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        # Annotators
        line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(width, height)) * 0.5)
        text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(width, height)) * 0.5

        label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
        trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)

        # st.video(source)

        if st.button("Execution"):
            with st.spinner("Running..."):
                st_frame = st.empty()
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break
                    
                    annotated_image = image.copy()
                    results = model.track(
                        source=image,
                        persist=True,
                        imgsz=640,
                        conf=confidence,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        # classes=class_filter,
                        retina_masks=True,
                        verbose=False
                    )[0]
                    detections = sv.Detections.from_ultralytics(results)
                    detections = detections.with_nms()

                    # Draw labels
                    object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
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
                        image=annotated_image,
                        caption='Output',
                        channels='BGR',
                        use_column_width=True )

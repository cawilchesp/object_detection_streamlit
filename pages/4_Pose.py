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
from tools.pose_annotator import PoseAnnotator
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink


# For debugging
from icecream import ic

st.set_page_config(layout="wide")

st.title('Pose Detection')

# SIDEBAR
st.sidebar.header("Model")

model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, config.POSE_MODEL_DICT[model_type])
else:
    st.error('Select Model in Sidebar')

# load pretrained DL model
try:
    model = config.load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
source_video = st.file_uploader(
    label="Choose a video..."
)
col1, col2, col3 = st.columns([2, 5, 2])
with col1:
    circles_activated = st.checkbox(label='Circles', value=True)
    lines_activated = st.checkbox(label='Lines', value=True)
    labels_activated = st.checkbox(label='Labels', value=False)
    line_thickness = int(st.slider("Select Model Confidence", 1, 50, 2))
with col2:
    st_frame = st.empty()
with col3:
    st.markdown('**Width**')
    width_text = st.markdown('0')
    st.markdown('**Height**')
    height_text = st.markdown('0')
    st.markdown('**Total Frames**')
    total_frames_text = st.markdown('0')
    st.markdown('**Frame Rate**')
    fps_text = st.markdown('0')

if source_video:
    if st.button("Execution"):
        with st.spinner("Running..."):
            try:
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(source_video.read())
                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                width_text.write(str(width))
                height_text.write(str(height))
                total_frames_text.write(str(total_frames))
                fps_text.write(str(fps))

                # Annotators
                pose_annotator = PoseAnnotator(thickness=line_thickness, radius=4)

                while cap.isOpened():
                    success, image = cap.read()
                    if success:
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
                        
                        # Draw poses
                        annotated_image = pose_annotator.annotate(
                            scene=image,
                            ultralytics_results=results,
                            circles=circles_activated,
                            lines=lines_activated,
                            labels=labels_activated
                        )     

                        st_frame.image(
                            annotated_image,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                        )
                    else:
                        cap.release()
                        break
            except Exception as e:
                st.error(f"Error loading video: {e}")
    
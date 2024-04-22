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

st.title('Video Detection')


# SIDEBAR
st.sidebar.header("DL Model Config")

model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

confidence = float(st.sidebar.slider("Select Model Confidence", 0.3, 1.0, 0.5))

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error('Select Model in Sidebar')

# load pretrained DL model
try:
    model = config.load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")


# image/video options
st.sidebar.header("Image/Video Config")

source_video = st.sidebar.file_uploader(
    label="Choose a video..."
)
col1, col2 = st.columns(2)
with col1:
    st_frame = st.empty()
with col2:
    # kpi1, kpi2, kpi3 = st.columns(3)
    # with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')
    # with kpi2:
        st.markdown('**Width**')
        kpi2_text = st.markdown('1')
    # with kpi3:
        st.markdown('**Height**')
        kpi3_text = st.markdown('2')

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

                kpi1_text.write(fps)
                kpi2_text.write(width)
                kpi3_text.write(height)

                # Annotators
                line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(width, height)) * 0.5)
                text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(width, height)) * 0.5

                label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
                bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
                trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)

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
                            # classes=class_filter,
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
                    else:
                        cap.release()
                        break
            except Exception as e:
                st.error(f"Error loading video: {e}")

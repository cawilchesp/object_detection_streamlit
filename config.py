from ultralytics import YOLO
import streamlit as st
from pathlib import Path
import sys


ROOT = Path('D:/Data')

# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]

# DL model config
DETECTION_MODEL_DIR = ROOT / 'models' / 'yolov8'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    'Nano',
    'Small',
    'Medium',
    'Large',
    'Extralarge' ]

# Pose model config
YOLOv8n_pose = DETECTION_MODEL_DIR / 'yolov8n-pose.pt'
YOLOv8s_pose = DETECTION_MODEL_DIR / 'yolov8s-pose.pt'
YOLOv8m_pose = DETECTION_MODEL_DIR / 'yolov8m-pose.pt'
YOLOv8l_pose = DETECTION_MODEL_DIR / 'yolov8l-pose.pt'
YOLOv8x_pose = DETECTION_MODEL_DIR / 'yolov8x-pose.pt'

POSE_MODEL_DICT = {
    'Nano': 'yolov8n-pose.pt',
    'Small': 'yolov8s-pose.pt',
    'Medium': 'yolov8m-pose.pt',
    'Large': 'yolov8l-pose.pt',
    'Extralarge': 'yolov8x-pose.pt'
}


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model
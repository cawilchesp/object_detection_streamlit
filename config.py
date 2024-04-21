from ultralytics import YOLO
import streamlit as st

# DL model config
DETECTION_MODEL_DIR = 'D:/Data/models'

DETECTION_MODEL_LIST = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt',
    'yolov9c.pt',
    'yolov9e.pt'
]

@st.cache_resource
def load_model(model_path: str):
    model = YOLO(model_path)
    return model
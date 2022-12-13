import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import torch

from PIL import Image

from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


def load_classes(source):
    with open(source, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def yolor_init():
    # --------------
    # YOLOR Settings
    # --------------
    cfg = './cfg/yolor_p6.cfg'
    weights = './yolor_p6.pt'

    device_opt = '0' # cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz = 1280

    # --------------
    # Initialization
    # --------------
    device = select_device(device_opt)
    half = device.type  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    model.half()  # to FP16

    return device, half, model

def detect(device, half, model, image, conf_thres):
    names = './data/coco.names'
    imgsz = 1280
    iou_thres = 0.5
    classes_opt = [0,1,2,3,5,7]
    # classes_opt = [0,1,2,3,5,7,14,15,16,17,18,19,20,21,22,23] # filter by class
    
    # Get names and colors
    names = load_classes(names)
    colors = {
        0 : [0, 0, 0],
        1 : [200, 200, 0],
        2 : [0, 0, 255],
        3 : [255, 0, 0],
        5 : [0, 150, 0],
        7 : [200, 0, 200]
    }

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # Padded resize
    img = letterbox(image, new_shape=imgsz, auto_size=64)[0]
            
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_opt, agnostic=False)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = image
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in det:
                x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                color = colors[int(cls)]

                # draw bbox on screen
                cv2.rectangle(im0, (x,y), (x+w,y+h), color, 1, cv2.LINE_AA)

                # draw class label on screen
                label = f'{names[int(cls)]} {conf:.2f}'
                t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
                cv2.rectangle(im0, (x,y), (x+t_size[0], y-t_size[1]-3), color, -1, cv2.LINE_AA)
                cv2.putText(im0, label, (x, y-2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)

    # Print time (inference + NMS)
    return im0, len(det), t2 - t1

# Main Page ------------------
st.title('Object Detection')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar ------------------
st.sidebar.title('Parameters')
app_mode = st.sidebar.selectbox('Choose the app mode', 
                                    ['Run on Image', 'Run on video', 'About App']
                                )

if app_mode == 'About App':
    # Main Page ------------------
    st.markdown('In this application we are using **YOLOR** for creating an object detection app.')

elif app_mode == 'Run on Image':
    # Sidebar ------------------
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    conf_thres = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    img_file_buffer = st.sidebar.file_uploader('Upload Image', type=['jpg','jpeg','png'])
    
    # Main Page ------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown('**Detected Objects**')
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Inference Time**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown('**Image Width**')
        kpi3_text = st.markdown('0')

    with kpi4:
        st.markdown('**Image Height**')
        kpi4_text = st.markdown('0')

#     # Processing ------------------
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Original image')
        st.sidebar.image(image)
        object_count = 0
        with torch.no_grad():
            device, half, model = yolor_init()
            out_image, object_count, inf_time = detect(device, half, model, image, conf_thres)

            kpi1_text.write(f'{object_count}')
            kpi2_text.write(f'{inf_time:.3f}')
            kpi3_text.write(f'{out_image.shape[1]}')
            kpi4_text.write(f'{out_image.shape[0]}')

            # Main Page ------------------
            st.subheader('Output Image')
            st.image(out_image, use_column_width = True)


elif app_mode == 'Run on video':
    # Sidebar ------------------
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    conf_thres = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    video_mode = st.sidebar.selectbox('Choose the video mode', 
                                    ['Video File', 'IP Address', 'Webcam']
                                )
    if video_mode == 'Video File':
        video_file_buffer = st.sidebar.file_uploader('Upload a Video', type = ['mp4', 'mov', 'avi', 'asf', 'm4v'])
    
    elif video_mode == 'IP Address':
        user = st.sidebar.text_input('User Name', value = 'Sier')
        password = st.sidebar.text_input('Password', value = 'RupuMaipo..2021*')
        camera_ip = st.sidebar.text_input('Camera IP')
        connect_ip = st.sidebar.button('Connect')

    elif video_mode == 'Webcam':
        use_webcam = st.sidebar.button('Use Webcam')

    # Main Page ------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown('**Detected Objects**')
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Inference Time**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown('**Image Width**')
        kpi3_text = st.markdown('0')

    with kpi4:
        st.markdown('**Image Height**')
        kpi4_text = st.markdown('0')

    # Processing ------------------
    vid = cv2.VideoCapture()
    if video_mode == 'Video File':
        tffile = tempfile.NamedTemporaryFile(delete=False)
        if video_file_buffer is not None:
            tffile.write(video_file_buffer.read())
            vid.open(tffile.name)
            st.sidebar.text('Input video')
            st.sidebar.video(tffile.name)

    elif video_mode == 'IP Address':
        if connect_ip:
            if camera_ip:
                source = f'rtsp://{user}:{password}@{camera_ip}/axis-media/media.amp?Transport=multicast'
                vid.open(source)
            else:
                st.sidebar.text('You need an IP address')
    elif video_mode == 'Webcam':
        if use_webcam:
            vid.open(0)

        
    if vid.isOpened():
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        device, half, model = yolor_init()
        
#     if stop_video:
#             vid.release()
        st.subheader('Output Image')
        stframe = st.empty()
        while(vid.isOpened()):
            ret, frame = vid.read()
            if not ret:
                st.sidebar.text('Video stopped !')
                break

            object_count = 0
            with torch.no_grad():
                out_image, object_count, inf_time = detect(device, half, model, frame, conf_thres)
                
                kpi1_text.write(f'{object_count}')
                kpi2_text.write(f'{inf_time:.3f}')
                kpi3_text.write(f'{out_image.shape[1]}')
                kpi4_text.write(f'{out_image.shape[0]}')

                # Main Page ------------------
                stframe.image(out_image, use_column_width = True)

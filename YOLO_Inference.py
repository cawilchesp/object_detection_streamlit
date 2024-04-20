import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import torch

from PIL import Image


# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Main Page ------------------
st.title('Object Detection')

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)


# app_mode = st.sidebar.selectbox('Choose the app mode', 
#                                     ['Run on Image', 'Run on video', 'About App']
#                                 )

# if app_mode == 'About App':
#     # Main Page ------------------
#     st.markdown('In this application we are using **YOLOR** for creating an object detection app.')

# elif app_mode == 'Run on Image':
#     # Sidebar ------------------
    
    
#     conf_thres = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    
#     img_file_buffer = st.sidebar.file_uploader('Upload Image', type=['jpg','jpeg','png'])
    
#     # Main Page ------------------
#     kpi1, kpi2, kpi3, kpi4 = st.columns(4)

#     with kpi1:
#         st.markdown('**Detected Objects**')
#         kpi1_text = st.markdown('0')

#     with kpi2:
#         st.markdown('**Inference Time**')
#         kpi2_text = st.markdown('0')

#     with kpi3:
#         st.markdown('**Image Width**')
#         kpi3_text = st.markdown('0')

#     with kpi4:
#         st.markdown('**Image Height**')
#         kpi4_text = st.markdown('0')

# #     # Processing ------------------
#     if img_file_buffer is not None:
#         image = np.array(Image.open(img_file_buffer))
#         st.sidebar.text('Original image')
#         st.sidebar.image(image)
#         object_count = 0
#         with torch.no_grad():
#             device, half, model = yolor_init()
#             out_image, object_count, inf_time = detect(device, half, model, image, conf_thres)

#             kpi1_text.write(f'{object_count}')
#             kpi2_text.write(f'{inf_time:.3f}')
#             kpi3_text.write(f'{out_image.shape[1]}')
#             kpi4_text.write(f'{out_image.shape[0]}')

#             # Main Page ------------------
#             st.subheader('Output Image')
#             st.image(out_image, use_column_width = True)


# elif app_mode == 'Run on video':
#     # Sidebar ------------------
    
#     conf_thres = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    
#     video_mode = st.sidebar.selectbox('Choose the video mode', 
#                                     ['Video File', 'IP Address', 'Webcam']
#                                 )
#     if video_mode == 'Video File':
#         video_file_buffer = st.sidebar.file_uploader('Upload a Video', type = ['mp4', 'mov', 'avi', 'asf', 'm4v'])
    
#     elif video_mode == 'IP Address':
#         user = st.sidebar.text_input('User Name', value = 'Sier')
#         password = st.sidebar.text_input('Password', value = 'RupuMaipo..2021*')
#         camera_ip = st.sidebar.text_input('Camera IP')
#         connect_ip = st.sidebar.button('Connect')

#     elif video_mode == 'Webcam':
#         use_webcam = st.sidebar.button('Use Webcam')

#     # Main Page ------------------
#     kpi1, kpi2, kpi3, kpi4 = st.columns(4)

#     with kpi1:
#         st.markdown('**Detected Objects**')
#         kpi1_text = st.markdown('0')

#     with kpi2:
#         st.markdown('**Inference Time**')
#         kpi2_text = st.markdown('0')

#     with kpi3:
#         st.markdown('**Image Width**')
#         kpi3_text = st.markdown('0')

#     with kpi4:
#         st.markdown('**Image Height**')
#         kpi4_text = st.markdown('0')

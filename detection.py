import streamlit as st
import torch
import PIL 
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def get_yolo_detection():
    
    st.markdown('''<h1 style="text-align: left; font-family: 'Gill Sans'; color: #D8D8D8"
            >Приложение по детекции автомобилей Тесла. Вы сможете определить: </h1><h1 style="text-align: left; font-family: 'Gill Sans'; color: #FF2A00"
            >CyberTruck, Model3, ModelS, ModelX, ModelY и Roadster</h1>''',
            unsafe_allow_html=True)


    model = torch.hub.load('ultralytics/yolov5', 'custom', path='our_models/best.pt')
    img_file = st.file_uploader('Choose photo', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
    
    
    if img_file:
        img = PIL.Image.open(img_file)
        results = model(img)
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(results.render()[0])
        st.pyplot(fig)

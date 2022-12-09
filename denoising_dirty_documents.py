import streamlit as st
from autoencoder_models import ConvAutoencoder
import torch
from PIL import Image
from torchvision import transforms as t 
import numpy as np

def clean_documents():

    tensor = t.ToTensor()
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('our_models/denoising-dirty-documents.pt', map_location=torch.device('cpu')))
    model.eval()


    def get_model_results(img):
        img = model(img.unsqueeze(0))[0][0].detach().cpu().numpy()
        img = np.where(img > .88, 1, img)
        img = np.where(img < .2, 0, img)
        return img


    def get_homology(preprocess_img, img, n):
        homologous_series = []
        img = img[0].detach().cpu().numpy()
        zeros = np.zeros(([n] + list(img.shape)))
        for i, t in enumerate(np.linspace(0., 1., n)):
            zeros[i] = img * (1-t) + preprocess_img * t
            homologous_series.append((torch.Tensor(zeros[i]).unsqueeze(0).detach().cpu() * 255).to(torch.uint8))
        return homologous_series


    st.markdown('''<h1 style="text-align: left; font-family: 'Gill Sans'; color: #D8D8D8"
            >Сервис по восстановлнию погибших документов</h1><h1 style="text-align: left; font-family: 'Gill Sans'; color: #FF2A00">спасите ваши записи</h1>''',
            unsafe_allow_html=True)


    img = st.file_uploader('Choose file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
    pic = st.slider("processing progress", 0, 10, 10)


    if img:
        img = Image.open(img)
        img = tensor(img)
        preprocess_img = get_model_results(img)
        homologous_series = get_homology(preprocess_img, img, 11)
        st.image(homologous_series[pic].detach().cpu().numpy()[0], use_column_width = True)

if __name__ == '__main__':
    clean_documents()
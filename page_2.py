import torch
import torch.nn as nn
import streamlit as st
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import os


def generate_number():

    image_size = 32
    batch_size = 16
    latent_size = 32

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(latent_size, 256, kernel_size=4, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )

            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True)            
            )

            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)            
            )

            self.upsample4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True)    
            )

            self.upsample5 = nn.Sequential(
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(1),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.upsample1(x)
            x = self.upsample2(x)
            x = self.upsample3(x)
            x = self.upsample4(x)
            x = self.upsample5(x)

            return x


    st.markdown('''<h1 style="text-align: left; font-family: 'Gill Sans'; color: #D8D8D8"
        >Сгенерируйте цифру от 0 до 9 </h1><h1 style="text-align: left; font-family: 'Gill Sans'; color: #FF2A00"
        >1 2 3 4 5 6 7 ...</h1>''',
        unsafe_allow_html=True)


    def load_generator():
        generator = Generator()
        #generator = Generator(pretrained=True)    
        generator.load_state_dict(torch.load('our_models/savegenerator.pt', map_location='cpu'))
        generator.eval()
        return generator

    def denorm(img_tensors):
        return img_tensors * 0.1307 + 0.3081

    def main():
        generator = load_generator()

        latent = torch.randn(1, 32, 1, 1)

        fake_images = denorm(generator(latent)).detach().cpu().numpy()
        fake_images = np.where(fake_images > .30, 1, fake_images)
        fake_images = np.where(fake_images < .29, 0, fake_images)

        gen_button = st.button("Generate")
        
        if gen_button:
            st.title("Generated number")
            fig = fake_images.squeeze(1)[0]
            st.image(fig, width = 250)
        
    main()
        
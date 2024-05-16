import io
from urllib.parse import urljoin
import zipfile
import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import numpy as np
import streamlit as lt
import pandas as pandas
import torch
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import requests
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

from Models.cdcgan_coloured_64_flip import cDCGAN_Generator_Colored_64_Flip
from Models.cdcgan_gray_128_flip import cDCGAN_Generator_Gray_128_Flip
from Models.cwgan_128_coloured import cWGAN_Generator_Colored_128
from Models.cwgan_128_gray import cWGAN_Generator_Gray_128
from Models.gan_genartor_coloured_128 import Gan_Generator_Colured
from Models.gan_generator_gray_128 import Gan_Generator_Gray
from Models.cwgan_64 import cWGAN_Generator_64

# STEPS:
# 1- put the path of the model
# 2- Put Your Gan_Generator_Colured Model
# 3- Put UR Z_DIM
# 4- Instantiate the Gan_generator_Colured
# 5- Move to GPU
# 6- Load the state dict
# 7-  Optionally move the model to GPU if available
# 8-  Set the model to evaluation mode
# 9-  Define a function to get noise
# 10- Define a function to show tensor images
# 11- Define a function to generate image
# 12- Generate an image using the model
# 13- Display the generated image


##############################################
# To save generated images by the model ######
##############################################


####

# 1-put the path of the model here

# GANs
#######################################
# "C:\\Users\\Seif Yasser\\Desktop\\Artificial intelligence\\Project\\a5r repo inshallah\\Models"
path_cGAN_Colored_128 = 'Trained-Models\\GAN_Colored_128_Gen_750.pth'
# path_cGAN_Gray_128 = 'Trained-Models\\GAN_Gray_gen_500_epochs.pth'


# DCGANs Flipped
#######################################
path_cDCGAN_Colored_Flip_64 = 'Trained-Models\\CDCGAN_Colored_64_Flipped_Gen_500.pth'
path_cDCGAN_Gray_Flip_128 = 'Trained-Models\\CDC_gen_450.pth'


# WGANs Gray
#######################################
path_CWGAN_Gray_Flip_64 = 'Trained-Models\\cWGAN_Gray_Scale_64_Flipped.pth'
path_CWGAN_Gray_No_Flip_128 = 'Trained-Models\\CWGAN_Gray_No_Flip_128_Gen_140.pth'


# WGANs Coloured
#######################################
path_CWGAN_Colored_Flip_128 = 'Trained-Models\\CWGAN_128_Flip_Coloured_gen_600.pth'
path_CWGAN_Colored_No_Flip_128 = 'Trained-Models\\CWGAN_128_No_Flip_Coloured_gen_375.pth'
path_CWGAN_Colored_Flip_64_lr_2e_4 = 'Trained-Models\\CWGAN_64_Flipped_Colored_lr_2e-4_gen_400.pth'
path_CWGAN_Colored_Flip_64_lr_5e_4 = 'Trained-Models\\CWGAN_64_Flipped_Colored_lr_5e-4_gen_700.pth'
path_CWGAN_Colored_No_Flip_64 = 'Trained-Models\\Gen_CWGAN_No_Flip_64_Colored_525.pth'


# Will be passed by the deployment as an argument with if condition with if selection from the user

# make torch on cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loading all models

gen_cGAN_Colored_128 = torch.load(path_cGAN_Colored_128)
# gen_cGAN_Gray_128 = torch.load(path_cGAN_Gray_128)

gen_cDCGAN_Colored_Flip_64 = torch.load(path_cDCGAN_Colored_Flip_64)
gen_cDCGAN_Gray_Flip_128 = torch.load(path_cDCGAN_Gray_Flip_128)

gen_WGAN_Gray_Flip_64 = torch.load(path_CWGAN_Gray_Flip_64)
gen_WGAN_Gray_No_Flip_128 = torch.load(path_CWGAN_Gray_No_Flip_128)

gen_CWGAN_Colored_Flip_128 = torch.load(path_CWGAN_Colored_Flip_128)
gen_CWGAN_Colored_No_Flip_128 = torch.load(path_CWGAN_Colored_No_Flip_128)
gen_CWGAN_Colored_Flip_64_lr_2e_4 = torch.load(
    path_CWGAN_Colored_Flip_64_lr_2e_4)
gen_CWGAN_Colored_Flip_64_lr_5e_4 = torch.load(
    path_CWGAN_Colored_Flip_64_lr_5e_4)
gen_CWGAN_Colored_No_Flip_64 = torch.load(path_CWGAN_Colored_No_Flip_64)

print("Models Loaded Successfully")


# Create objects of each model

gen_cGAN_Colored_128_Model = Gan_Generator_Colured(
    64)
gen_cGAN_Colored_128_Model.load_state_dict(gen_cGAN_Colored_128)
if torch.cuda.is_available():
    gen_cGAN_Colored_128_Model.cuda()
gen_cGAN_Colored_128_Model.eval()
# gen_cGAN_Gray_128 = Gan_Generator_Gray(64).load_state_dict(gen_cGAN_Gray_128) -> error

gen_cDCGAN_Colored_Flip_64_Model = cDCGAN_Generator_Colored_64_Flip(
    100, 3, 64, 3, 64, 100)
gen_cDCGAN_Colored_Flip_64_Model.load_state_dict(
    gen_cDCGAN_Colored_Flip_64)
gen_cDCGAN_Colored_Flip_64_Model.eval()

gen_cDCGAN_Gray_Flip_128_Model = cDCGAN_Generator_Gray_128_Flip(
    100, 3, 64, 3, 128, 100)
gen_cDCGAN_Gray_Flip_128_Model.load_state_dict(gen_cDCGAN_Gray_Flip_128)
gen_cDCGAN_Gray_Flip_128_Model.eval()


gen_WGAN_Gray_Flip_64_Model = cWGAN_Generator_64(
    100, 3, 64, 3, 64, 100)
gen_WGAN_Gray_Flip_64_Model.load_state_dict(
    gen_WGAN_Gray_Flip_64)
gen_WGAN_Gray_Flip_64_Model.eval()

gen_WGAN_Gray_No_Flip_128_Model = cWGAN_Generator_Gray_128(
    100, 3, 64, 3, 128, 100)
gen_WGAN_Gray_No_Flip_128_Model.load_state_dict(gen_WGAN_Gray_No_Flip_128)
gen_WGAN_Gray_No_Flip_128_Model.eval()


gen_CWGAN_Colored_Flip_128_Model = cWGAN_Generator_Colored_128(
    100, 3, 64, 3, 128, 100)
gen_CWGAN_Colored_Flip_128_Model.load_state_dict(
    gen_CWGAN_Colored_Flip_128)
gen_CWGAN_Colored_Flip_128_Model.eval()

gen_CWGAN_Colored_No_Flip_128_Model = cWGAN_Generator_Colored_128(
    100, 3, 64, 3, 128, 100)
gen_CWGAN_Colored_No_Flip_128_Model.load_state_dict(
    gen_CWGAN_Colored_No_Flip_128)
gen_CWGAN_Colored_No_Flip_128_Model.eval()


gen_CWGAN_Colored_Flip_64_lr_2e_4_Model = cWGAN_Generator_64(
    100, 3, 64, 3, 64, 100)
gen_CWGAN_Colored_Flip_64_lr_2e_4_Model.load_state_dict(
    gen_CWGAN_Colored_Flip_64_lr_2e_4)
gen_CWGAN_Colored_Flip_64_lr_2e_4_Model.eval()


gen_CWGAN_Colored_Flip_64_lr_5e_4_Model = cWGAN_Generator_64(
    100, 3, 64, 3, 64, 100)
gen_CWGAN_Colored_Flip_64_lr_5e_4_Model.load_state_dict(
    gen_CWGAN_Colored_Flip_64_lr_5e_4)
gen_CWGAN_Colored_Flip_64_lr_5e_4_Model.eval()


gen_CWGAN_Colored_No_Flip_64_Model = cWGAN_Generator_64(
    100, 3, 64, 3, 64, 100)
gen_CWGAN_Colored_No_Flip_64_Model.load_state_dict(
    gen_CWGAN_Colored_No_Flip_64)
gen_CWGAN_Colored_No_Flip_64_Model.eval()

z_dim_2 = 64
z_dim = 100


def get_noise(no_outputs, z_dim, device='cuda'):
    return torch.randn(no_outputs, z_dim, 1, 1)


def get_noise_2(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, device=device)


# fake_noise=get_noise(2,100)


def generate_image(model, gans, num_outputs, label=None):
    """
    Generate an image using the given model.

    Args:
      model (torch.nn.Module): The generative model.
      label (int, optional): The label to generate the image for. If None, generate an image without a label.

    Returns:
      PIL.Image.Image: The generated image.
    """
    # model.eval()
    if gans:
        z_dim = 64
        fake_noise = get_noise_2(96, 64)
        print("Fake Noise")
        print(fake_noise)
        print(fake_noise.shape)
        with torch.no_grad():
            generated_image = model(fake_noise)
    else:
        z_dim = 100
        fake_noise = get_noise(num_outputs, z_dim)
        with torch.no_grad():
            generated_image = model(fake_noise, label[:])

    return generated_image


##########################################
transform = T.ToPILImage(mode='RGB')  # Convert a tensor to an image

# Generate an image using the model
# generated_image = generate_image(model)

# Display the generated image
# st.image(generated_image, caption='Generated Image')


def configure_sidebar() -> None:
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Yo fam! Start here ↓**", icon="👋🏾")
            with st.expander(":primary[**Refine your output here**]"):
                num_outputs = st.slider(
                    "Number of images to Generate", value=1, min_value=1, max_value=64)

                refine = st.selectbox(
                    "Select GANs Model to use", ("GAN-C", "cDCGAN-64-C", "cDCGAN-128-NC", "CWGAN-64-F-NC", "CWGAN-64-F-C", "CWGAN-64-C", "CWGAN-64-NC", "CWGAN-128-F-C", "CWGAN-128-C", "CWGAN-128-NC"))
                scheduler = st.selectbox(
                    'Image to Generate', ('Boot', 'Sandal', 'Shoe'))

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Generate", type="primary", use_container_width=True)

        return submitted, num_outputs, scheduler, refine


def main_page(submitted: bool, num_outputs: int,
              scheduler: str,  refine: str) -> None:
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your Choices into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            # try:
            # Only call the API if the "Submit" button was pressed
            if submitted:
                all_images = []  # List to store all generated images
                label = torch.tensor([0]).repeat(num_outputs)
                if scheduler == "Boot":
                    label = torch.tensor([0]).repeat(num_outputs)
                elif scheduler == "Sandal":
                    label = torch.tensor([1]).repeat(num_outputs)
                else:
                    label = torch.tensor([2]).repeat(num_outputs)

                # "GAN-C","GAN-NC","cDCGAN-64-C","cDCGAN-128-NC","CWGAN-64-F-NC","CWGAN-64-F-C","CWGAN-64-C","CWGAN-64-NC", "CWGAN-128-F-C","CWGAN-128-C","CWGAN-128-NC"
                # z_dim = 100
                num_outputs = int(num_outputs)
                flag = False
                # if refine == "GAN-C":
                #     z_dim = 64
                #     model = gen_cGAN_Colored_128_Model
                #     flag = True
                # # elif refine=="GAN-NC":
                # #     z_dim=64
                # #     model = gen_cGAN_Gray_128
                # else:
                #     z_dim = 100
                #     # gans=False

                # model = gen_cDCGAN_Colored_Flip_64
                if refine == "GAN-C":
                    model = gen_cGAN_Colored_128_Model
                    flag = True
                elif refine == "cDCGAN-64-C":
                    model = gen_cDCGAN_Colored_Flip_64_Model
                elif refine == "cDCGAN-128-NC":
                    model = gen_cDCGAN_Gray_Flip_128_Model
                elif refine == "CWGAN-64-F-NC":
                    model = gen_WGAN_Gray_Flip_64_Model
                elif refine == "CWGAN-64-F-C":
                    # gen_CWGAN_Colored_Flip_64_lr_2e_4_Model
                    model = gen_CWGAN_Colored_Flip_64_lr_2e_4_Model
                elif refine == "CWGAN-64-C":
                    model = gen_CWGAN_Colored_No_Flip_64_Model
                elif refine == "CWGAN-64-NC":
                    model = gen_WGAN_Gray_Flip_64_Model
                elif refine == "CWGAN-128-F-C":
                    model = gen_CWGAN_Colored_Flip_128_Model
                elif refine == "CWGAN-128-C":
                    model = gen_CWGAN_Colored_No_Flip_128_Model
                elif refine == "CWGAN-128-NC":
                    model = gen_WGAN_Gray_No_Flip_128_Model

                output = generate_image(model, flag, num_outputs, label)
                # output = output[0]
                if refine == "GAN-C":
                    output = output.cpu()

                img_grid = make_grid(output[:num_outputs], normalize=True)
                plt.imshow(img_grid.permute(1, 2, 0))
                plt.savefig('output.png')
                plt.close()
                # read that image downloaded
                output = Image.open('output.png')
                # show it in streamlit
                st.image(output, caption="Generated Image",
                         use_column_width=False)
                if output:
                    st.toast(
                        'Your image has been generated!', icon='😍')
                    st.session_state.generated_image = output

    else:
        pass


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted, num_outputs, scheduler, refine = configure_sidebar()
    main_page(submitted, num_outputs, scheduler, refine)


if __name__ == "__main__":
    main()

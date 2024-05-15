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
path_cGAN_Colored_128 = 'Trained-Models\GAN_Colored_128_Gen_750.pth'
path_cGAN_Gray_128 = 'Trained-Models\GAN_Gray_gen_500_epochs.pth'


# DCGANs Flipped
#######################################
path_cDCGAN_Colored_Flip_64='Trained-Models\CDCGAN_Colored_64_Flipped_Gen_500.pth'
path_cDCGAN_Gray_Flip_128='Trained-Models\CDCGANs_Gray_Scale_128_500.pth'


# WGANs Gray
#######################################
path_CWGAN_Gray_Flip_64='Trained-Models\cWGAN_Gray_Scale_64_Flipped.pth'
# path_CWGAN_Gray_No_Flip_128=''


# WGANs Coloured
#######################################
path_CWGAN_Colored_Flip_128='Trained-Models\CWGAN_128_Flip_Coloured_gen_600.pth'
path_CWGAN_Colored_No_Flip_128='Trained-Models\CWGAN_128_No_Flip_Coloured_gen_375.pth'
path_CWGAN_Colored_Flip_64_lr_2e_4='Trained-Models\CWGAN_64_Flipped_Colored_lr_2e-4_gen_400.pth'
path_CWGAN_Colored_Flip_64_lr_5e_4='Trained-Models\CWGAN_64_Flipped_Colored_lr_5e-4_gen_700.pth'
path_CWGAN_Colored_No_Flip_64='Trained-Models\Gen_CWGAN_No_Flip_64_Colored_525.pth'



# Will be passed by the deployment as an argument with if condition with if selection from the user

# make torch on cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loading all models

gen_cGAN_Colored_128 = torch.load(path_cGAN_Colored_128)
gen_cGAN_Gray_128 = torch.load(path_cGAN_Gray_128)

gen_cDCGAN_Colored_Flip_64 = torch.load(path_cDCGAN_Colored_Flip_64)
gen_cDCGAN_Gray_Flip_128 = torch.load(path_cDCGAN_Gray_Flip_128)

gen_WGAN_Gray_Flip_64 = torch.load(path_CWGAN_Gray_Flip_64)
# gen_WGAN_Gray_No_Flip_128 = torch.load(path_CWGAN_Gray_No_Flip_128)

gen_CWGAN_Colored_Flip_128 = torch.load(path_CWGAN_Colored_Flip_128)
gen_CWGAN_Colored_No_Flip_128 = torch.load(path_CWGAN_Colored_No_Flip_128)
gen_CWGAN_Colored_Flip_64_lr_2e_4 = torch.load(path_CWGAN_Colored_Flip_64_lr_2e_4)
gen_CWGAN_Colored_Flip_64_lr_5e_4 = torch.load(path_CWGAN_Colored_Flip_64_lr_5e_4)
gen_CWGAN_Colored_No_Flip_64 = torch.load(path_CWGAN_Colored_No_Flip_64)

print("Models Loaded Successfully")


# Create objects of each model

gen_cGAN_Colored_128 = Gan_Generator_Colured(64).load_state_dict(gen_cGAN_Colored_128)
# gen_cGAN_Gray_128 = Gan_Generator_Gray(64).load_state_dict(gen_cGAN_Gray_128) -> error

gen_cDCGAN_Colored_Flip_64 = cDCGAN_Generator_Colored_64_Flip(100, 3, 64, 3, 64, 100).load_state_dict(gen_cDCGAN_Colored_Flip_64)
# gen_cDCGAN_Gray_Flip_128 = cDCGAN_Generator_Gray_128_Flip(100, 3, 64, 3, 128, 100).load_state_dict(gen_cDCGAN_Gray_Flip_128) -> error

gen_WGAN_Gray_Flip_64 = cWGAN_Generator_64(100, 3, 64, 3, 64, 100).load_state_dict(gen_WGAN_Gray_Flip_64)
# gen_WGAN_Gray_No_Flip_128 = cWGAN_Generator_Gray_128(100, 3, 64, 3, 128, 100).load_state_dict(gen_WGAN_Gray_No_Flip_128) -> Missing Path

gen_CWGAN_Colored_Flip_128 = cWGAN_Generator_Colored_128(100, 3, 64, 3, 128, 100).load_state_dict(gen_CWGAN_Colored_Flip_128)
gen_CWGAN_Colored_No_Flip_128 = cWGAN_Generator_Colored_128(100, 3, 64, 3, 128, 100).load_state_dict(gen_CWGAN_Colored_No_Flip_128)

gen_CWGAN_Colored_Flip_64_lr_2e_4 = cWGAN_Generator_64(100, 3, 64, 3, 64, 100).load_state_dict(gen_CWGAN_Colored_Flip_64_lr_2e_4)
gen_CWGAN_Colored_Flip_64_lr_5e_4 = cWGAN_Generator_64(100, 3, 64, 3, 64, 100).load_state_dict(gen_CWGAN_Colored_Flip_64_lr_5e_4)

gen_CWGAN_Colored_No_Flip_64 = cWGAN_Generator_64(100, 3, 64, 3, 64, 100).load_state_dict(gen_CWGAN_Colored_No_Flip_64)


z_dim_2 = 64
z_dim = 100


def get_noise( z_dim, device='cuda'):
    return torch.randn( z_dim, device=device)


# fake_noise=get_noise(2,100)



def generate_image(model,fake_noise,label = None):
    
    """
    Generate an image using the given model.

    Args:
      model (torch.nn.Module): The generative model.
      label (int, optional): The label to generate the image for. If None, generate an image without a label.

    Returns:
      PIL.Image.Image: The generated image.
    """
    
    if label is None:
        with torch.no_grad():
            generated_image = model(fake_noise)
    
    else:
        with torch.no_grad():
            generated_image = model(fake_noise, label)
    
    return generated_image


##########################################


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
                    "Number of images to Generate", value=1, min_value=1, max_value=4)
                
                # Colored = st.selectbox(
                #     'Colored?', ('YES', 'NO'))
                # if Colored=="YES":
                    # refine = st.selectbox(
                    # "Select GANs Model to use", ("GAN","cDCGAN-64","CWGAN-64-F","CWGAN-64", "CWGAN-128-F","CWGAN-128"))
                # else:
                refine = st.selectbox(
                    "Select GANs Model to use", ("GAN-C","GAN-NC","cDCGAN-64-C","cDCGAN-128-NC","CWGAN-64-F-NC","CWGAN-64-F-C","CWGAN-64-C","CWGAN-64-NC", "CWGAN-128-F-C","CWGAN-128-C","CWGAN-128-NC"))
                
                if refine=="GAN-C" or refine=="GAN-NC":
                    scheduler=None
                else:
                    scheduler = st.selectbox(
                        'Image to Generate', ('Shoe', 'Boot', 'Sandal'))
                                   

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
                # Calling the replicate API to get the image
                # with generated_images_placeholder.container():
                all_images = []  # List to store all generated images

                if scheduler == "Shoe":
                    label = torch.tensor([0]).repeat(num_outputs)
                elif scheduler == "Boot":
                    label = torch.tensor([1]).repeat(num_outputs)
                else:
                    label = torch.tensor([2]).repeat(num_outputs)
                

                # "GAN-C","GAN-NC","cDCGAN-64-C","cDCGAN-128-NC","CWGAN-64-F-NC","CWGAN-64-F-C","CWGAN-64-C","CWGAN-64-NC", "CWGAN-128-F-C","CWGAN-128-C","CWGAN-128-NC"
                if refine=="GAN-C":
                    z_dim=64
                    model = gen_cGAN_Colored_128
                    # gans=True
                # elif refine=="GAN-NC":
                #     z_dim=64
                #     model = gen_cGAN_Gray_128
                    # gans=True
                else:
                    z_dim=100
                    # gans=False
                
                if refine=="cDCGAN-64-C":
                    model = gen_cDCGAN_Colored_Flip_64
                elif refine=="cDCGAN-128-NC":
                    model = gen_cDCGAN_Gray_Flip_128
                elif refine=="CWGAN-64-F-NC":
                    model = gen_WGAN_Gray_Flip_64
                elif refine=="CWGAN-64-F-C":
                    model = gen_CWGAN_Colored_Flip_64_lr_2e_4
                elif refine=="CWGAN-64-C":
                    model = gen_CWGAN_Colored_No_Flip_64
                elif refine=="CWGAN-64-NC":
                    model = gen_WGAN_Gray_Flip_64
                elif refine=="CWGAN-128-F-C":
                    model = gen_CWGAN_Colored_Flip_128
                elif refine=="CWGAN-128-C":
                    model = gen_CWGAN_Colored_No_Flip_128
                # elif refine=="CWGAN-128-NC":
                #     model = gen_CWGAN_Colored_Flip_128
                                

                    

                fake_noise=get_noise(z_dim)

                output = generate_image(model,fake_noise,label)
                output = output[0]
                output = output.cpu()
                # output = transform(output)

                if output:
                    st.toast(
                        'Your image has been generated!', icon='😍')
                    # Save generated image to session state
                    st.session_state.generated_image = output

                    # Displaying the image
                    # for image in st.session_state.generated_image:
                    with st.container():

                        st.image(
                            output, caption="Generated Image 🎈", use_column_width=True)
                        # Add image to the list
                        all_images.append(output)
                        # try:
                        # print("Seif")
                        # response = "https://"+requests.get(output)
                        # except:
                        #     pass
                        # # Convert the image data to a PIL image
                        # pil_image = Image.open(
                        #     io.BytesIO(response.content))

                        # Display the PIL image using show_tensor_images()
                        # show_tensor_images(ToTensor()(pil_image))
                # Save all generated images to session state
                # list1 = []
                # list1.append(output)
                st.session_state.all_images = all_images

                # Create a BytesIO object
                zip_io = io.BytesIO()

                # Download option for each image
                with zipfile.ZipFile(zip_io, 'w') as zipf:
                    for i, image in enumerate(st.session_state.all_images):
                        response = requests.get("https://")
                        absolute_url = urljoin(response.url, image)
                        response = requests.get(absolute_url)
                        print(response.status_code)
                        print(response)
                        if response.status_code == 200:
                            image_data = response.content
                            # Write each image to the zip file with a name
                            zipf.writestr(
                                f"output_file_{i+1}.png", image_data)
                        else:
                            st.error(
                                f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                # Create a download button for the zip file
                st.download_button(
                    ":red[**Download All Images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
            status.update(label="✅ Images generated!",
                          state="complete", expanded=False)
            # except Exception as e:
            #     print(e)
            #     st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    # with gallery_placeholder.container():
    #     img = image_select(
    #         label="Like what you see? Right-click and save! It's not stealing if we're sharing! 😉",
    #         images=[
    #             "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
    #             "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
    #             "gallery/cheetah.png", "gallery/viking.png",
    #         ],
    #         captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
    #                   "An astronaut riding a rainbow unicorn, cinematic, dramatic",
    #                   "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
    #                   "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
    #                   "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
    #                   "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
    #                   "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
    #                   ],
        # use_container_width=True
        # )


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

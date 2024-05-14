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

# STEPS:
# 1-put the path of the model
# 2-Put Your Generator Model
# 3- Put UR Z_DIM
# 4- Instantiate the generator
# 5- Move to GPU
# 6- Load the state dict
# 7- Optionally move the model to GPU if available
# 8- Set the model to evaluation mode
# 9- Define a function to get noise
# 10- Define a function to show tensor images
# 11- Define a function to generate image
# 12- Generate an image using the model
# 13- Display the generated image


##############################################
# To save generated images by the model ######
##############################################


####

# 1-put the path of the model here
path_CGAN_Colored = "C:\\Users\\Seif Yasser\\Desktop\\Artificial intelligence\\Project\\local-repo\\Image-Generation-Using-Generative-AI\\cGANs-Colored\\Architecture 1\\gen_500_epochs.pth"
path_CGAN_Gray = ""

path_DCGAN_Colored = ""
path_DCGAN_Gray = ""

path_WGAN_Colored = ""
path_WGAN_Gray = ""

# Will be passed by the deployment as an argument with if condition with if selection from the user
model = torch.load(path_CGAN_Colored)

# 2-Put Your Generator Model Here


class Generator(torch.nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 16384),  # Output size to match 128x128 image
            nn.Tanh()
        )

    def forward(self, x):
        # Reshape to N x 1 x 128 x 128
        return self.gen(x).view(-1, 3, 128, 128)


# 3- Put UR Z_DIM HERE
z_dim = 64
# 4- Instantiate the generator
gen = Generator(z_dim)

# 5- Move to GPU "DO NOT CHANGE"
gen = gen.to('cuda')


# Load the state dict "DO NOT CHANGE"
my_model = Generator(z_dim)
# load the state dict "DO NOT CHANGE"
my_model.load_state_dict(model)
# Optionally move the model to GPU if available
if torch.cuda.is_available():
    my_model.cuda()
# Set the model to evaluation mode
my_model.eval()


def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, device=device)


# Will be Changed if the Images are Gray-Scaled
def show_tensor_images(image_tensor, num_images=25, size=(3, 128, 128)):
    print(image_tensor.shape)
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.show()


def generate_image(model):
    """
    Generate an image using the given model.

    Args:
      model (torch.nn.Module): The generative model.

    Returns:
      PIL.Image.Image: The generated image.
    """

    # Generate an image from the noise vector
    with torch.no_grad():
        generated_image = my_model(fake_noise)

    # Remove the extra dimension from the generated image tensor
    # show_tensor_images(generated_image)
    # generated_image = generated_image.squeeze()

    # Convert the generated image tensor to a PIL image
    # generated_image = transforms.ToPILImage()(generated_image)

    return generated_image


fake_noise = get_noise(96, 64).to('cuda')

fake = my_model(fake_noise)
# len(fake)
# show_tensor_images(fake)
output = generate_image(my_model)
# type(output)
# output.shape
img = output[0]
transform = T.ToPILImage()
# img.shape
save_image(img, 'img1generated.png')
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
                scheduler = st.selectbox(
                    'Image to Generate', ('Shoe', 'Boot', 'Sandal'))
                Colored = st.selectbox(
                    'Colored?', ('YES', 'NO'))
                refine = st.selectbox(
                    "Select GANs Model to use)", ("GAN", "CGAN", "CDCGAN", "CWGAN"))

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Generate", type="primary", use_container_width=True)

        return submitted, num_outputs, scheduler, Colored, refine


def main_page(submitted: bool, num_outputs: int,
              scheduler: str, Colored: str, refine: str,) -> None:
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
                output = generate_image(model)
                output = output[0]
                output = output.cpu()
                output = transform(output)

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
    submitted, num_outputs, scheduler, Colored, refine = configure_sidebar()
    main_page(submitted, num_outputs, scheduler, Colored, refine)


if __name__ == "__main__":
    main()

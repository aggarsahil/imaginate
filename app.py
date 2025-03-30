import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from authtoken import auth_token  # Ensure authtoken.py exists

# Load Stable Diffusion Model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    revision="fp16", 
    allow_pickle=False, 
    use_auth_token=auth_token
)
pipe.to(device)

# Function to generate image
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio UI
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter Prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Imaginate - AI Image Generator",
    description="Enter a text prompt and generate AI art using Stable Diffusion."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

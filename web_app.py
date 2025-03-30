from flask import Flask, request, render_template, send_file
import torch
from diffusers import StableDiffusionPipeline
from authtoken import auth_token  # Ensure authtoken.py exists
from PIL import Image

app = Flask(__name__)

# Load Model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    revision="fp16",
    allow_pickle=False, 
    use_auth_token=auth_token
)
pipe.to(device)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form["prompt"]
        image = pipe(prompt).images[0]
        image_path = "static/generated.png"
        image.save(image_path)
        return render_template("index.html", image_url=image_path)

    return render_template("index.html", image_url=None)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import torch
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

app = Flask(__name__)

# Load model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    allow_pickle=False,
    use_auth_token=auth_token
)
pipe.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    image = pipe(prompt).images[0]
    image.save('static/generatedimage.png')

    return 'Image generated successfully!'

if __name__ == '__main__':
    app.run(debug=True)

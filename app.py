import tkinter as tk
import customtkinter as ctk  # Fixed import
from PIL import Image, ImageTk
from authtoken import auth_token  # Ensure authtoken.py exists

import torch
from diffusers import StableDiffusionPipeline

# Initialize the app
app = ctk.CTk()  # Use CTk, not tk.Tk()
app.geometry("532x632")
app.title("Imaginate")
ctk.set_appearance_mode("dark")

# Widgets
prompt = ctk.CTkEntry(
    master=app,
    height=40,
    width=512,
    font=("Arial", 20),
    text_color="black",
    fg_color="white"
)
prompt.pack(pady=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
lmain.pack()

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

def generate():
    image = pipe(prompt.get()).images[0]  # Updated syntax
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep reference!

trigger = ctk.CTkButton(
    master=app,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    text="Generate",
    command=generate
)
trigger.pack()

app.mainloop()

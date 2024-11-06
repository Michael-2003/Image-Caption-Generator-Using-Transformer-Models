import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np
from utility import *
from model import *
from parameters import *
from caption_generator import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Load the necessary models and vectorization layers
# Make sure these are loaded only once to avoid redundant computation

# Global variables to store the selected image path and the loaded image
image_path = None
img = None
photo_img = None

def select_image():
    global image_path, img, photo_img
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        img = Image.open(image_path)
        img = img.resize((400, 400))
        photo_img = ImageTk.PhotoImage(img)  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
        caption_text.delete(1.0, tk.END)

def generate_caption_button():
    if not image_path:
        messagebox.showerror("Error", "Please select an image first")
        return

    caption = generate_caption(image_path)
    caption_text.delete(1.0, tk.END)
    caption_text.insert(tk.END, caption)

def reset():
    global image_path, img, photo_img
    image_path = None
    img = None
    photo_img = None
    canvas.delete("all")
    caption_text.delete(1.0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Image Caption Generator")
root.geometry("700x700")

# Add a label
label = tk.Label(root, text="Select an Image and Generate a Caption")
label.pack(pady=10)

# Create a canvas to display the image
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# Add buttons to select an image and generate a caption
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(pady=5)

btn_generate = tk.Button(root, text="Generate Caption", command=generate_caption_button)
btn_generate.pack(pady=5)

# Add a button to reset the image and caption
btn_reset = tk.Button(root, text="Reset", command=reset)
btn_reset.pack(pady=5)

# Add a text box to display the generated caption
caption_text = tk.Text(root, height=4, width=50)
caption_text.pack(pady=10)

root.mainloop()

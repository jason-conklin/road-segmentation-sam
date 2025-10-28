import ipyleaflet as L
import os
from faicons import icon_svg
from geopy.distance import geodesic, great_circle
from shared import BASEMAPS, CITIES
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget
import ipywidgets as widgets
from IPython.display import display, Image, HTML

# Add sidewalk segmentation related imports
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
#from model import SidewalkSegmentationModel

# Initialize sidewalk segmentation model
#model = SidewalkSegmentationModel()
#model.eval()

# Define function to perform sidewalk segmentation on uploaded image
def perform_sidewalk_segmentation(image_path):
    # Load and preprocess the uploaded image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_image = transform(image).unsqueeze(0)

    # Perform sidewalk segmentation
    with torch.no_grad():
        output = model(input_image)
        segmented_image = np.argmax(output[0].cpu().numpy(), axis=0)

    return segmented_image

# Define UI elements
ui.page_opts(title="Sidewalk Segmentation", fillable=True)

with ui.sidebar():
    # Add upload image widget
    ui.input_file("upload_image", "Upload Image")

# Define UI layout
with ui.layout_column_wrap(fill=False):
    with ui.card():
        ui.card_header("Segmented Image")

        @render_widget
        def segmented_image_widget():
            return L.Map(zoom=4, center=(0, 0))

# Reactive value to store uploaded image path
uploaded_image_path = reactive.value()

# Update the reactive value when the image is uploaded
@reactive.effect
def _():
    uploaded_image_path.set(input.upload_image())

# Update segmented image widget when new image is uploaded
@reactive.effect
def _():
    if uploaded_image_path() is not None:
        segmented_image = perform_sidewalk_segmentation(uploaded_image_path())
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(segmented_image.astype(np.uint8))
        # Define bounds of the image overlay
        bounds = [[-90, -180], [90, 180]]  # Adjust as needed
        # Display segmented image on the map
        segmented_image_layer = L.ImageOverlay(url=pil_image, bounds=bounds)
        segmented_image_widget().add_layer(segmented_image_layer)

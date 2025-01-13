from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import geemap
import ee
import os
import shutil
import numpy as np
import pandas as pd
import rasterio
import pickle
import matplotlib.pyplot as plt
import random
from random import randint
import earthpy.plot as ep

# FastAPI initialization
app = FastAPI()
number = str(randint(1, 1000))

# Earth Engine initialization
ee.Authenticate()
ee.Initialize(project='ee-laxminarayan090503')

# Set up static files and templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create 'uploaded_files' directory if it doesn't exist
uploaded_files_dir = "uploaded_files"
os.makedirs(uploaded_files_dir, exist_ok=True)

# Function to convert TIFF to DataFrame
def tiff_to_csv(image):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    data = [image.read(i).flatten() for i in range(1, image.count + 1)]
    dataset = pd.DataFrame(np.array(data).T, columns=bands)
    dataset.fillna(0.0001, inplace=True)
    return dataset

# Function to plot classified map
def plot_classified_map(image, labels, image_shape, title, save_path='static/classified_map.png'):
    color_map = {
        'Water': [0.53, 0.81, 0.98],
        'BuiltUp': [1, 0, 0],
        'Vegetation': [0.13, 0.55, 0.13],
        'BarrenLand': [1, 0.65, 0],
        'Agricultural': [1, 1, 0]
    }

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    
    # RGB Image Bands 4, 3, 2
    image_vis_432 = np.stack([image.read(b) for b in [4, 3, 2]])
    ep.plot_rgb(image_vis_432, ax=axs[0], stretch=True)
    axs[0].set_title('Image (Bands 4, 3, 2)')
    axs[0].axis('off')

    # RGB Image Bands 8, 4, 3
    image_vis_843 = np.stack([image.read(b) for b in [8, 4, 3]])
    ep.plot_rgb(image_vis_843, ax=axs[1], stretch=True)
    axs[1].set_title('Image (Bands 8, 4, 3)')
    axs[1].axis('off')

    # Classified land cover map
    land_cover = labels.reshape(image_shape)
    rgb_array = np.array([[color_map.get(value, [0, 0, 0]) for value in row] for row in land_cover])
    axs[2].imshow(rgb_array)
    axs[2].set_title(title)
    axs[2].axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return os.path.abspath(save_path)

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "image_url": None})

@app.post("/classify/")
async def classify_image(request: Request, file: UploadFile = File(...), method: str = Form(...)):
    temp_file_path = os.path.join(uploaded_files_dir, f"temp_image_{number}.tif")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Open and process TIFF image
    image = rasterio.open(temp_file_path)
    image_dataset = tiff_to_csv(image)
    image_shape = image.read(1).shape

    # Load pre-trained model
    model_file = f"classification_methods/{method}.pkl"
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            clf = pickle.load(file)
    else:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": "Model not found."})

    # Predict and classify image
    labels = clf.predict(image_dataset)
    saved_image_path = plot_classified_map(image, labels, image_shape, "Classified Land Cover Map")
    image_url = os.path.basename(saved_image_path)
    
    return templates.TemplateResponse("index.html", {"request": request, "image_url": image_url})

@app.post("/download/")
async def download_tiff(coordinates: str = Form(...), file_name: str = Form(...)):
    geom = ee.Geometry.Polygon(eval(coordinates))
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1) \
        .filterDate('2023-01-01', '2023-12-31') \
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']) \
        .median().clip(geom).multiply(0.0001)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_name,
        scale=10,
        region=geom,
        fileFormat='GeoTIFF',
        maxPixels=1e13,
        folder='LULC',
        fileNamePrefix=file_name
    )
    task.start()

    return {"message": "Download initiated. Check your Google Drive.", "file_name": file_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1200)

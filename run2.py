import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
import torch
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from utility_functions_2 import *
import argparse
import os 
torch.cuda.empty_cache()
from pathlib import Path
import cv2

from object_remover import clean_background

from rotate_image import flip_image

from crop_images import crop_image

from image_overlaying import image_overlay

def main():
    parser = argparse.ArgumentParser(description='Segment an object in an image')
    parser.add_argument('--image', required=True, help='Path to the input image file.')
    parser.add_argument('--object', required=True, help='Class name to process.')
    parser.add_argument('--output', required=True, help='Path to save the output image.')
    
    args = parser.parse_args()
    
    
    image_path = args.image
    labels = args.object
    output_path= args.output
    object=args.object
    folder_path=Path.cwd()
    import os

# Define the folder names
    print(folder_path)
    folder_name = str(folder_path)+ f"/{object}/"
    
    os.mkdir(folder_name)
    

    # Check if 'folder' exists
    
        
        # Create the new unique folder


    
    
    print("folder with object name created")
    threshold = 0.3

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    image_array, detections = grounded_segmentation(
        image=image_path,
        labels=[labels],
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    bbox=plot_detections_mask(image_array, detections,folder_name)
    print(bbox)
    



    
    
    
    
    
    clean_background(image_path,folder_name+'rect_mask.png',folder_name+'clean_background.jpg')
    
    rotated,horizontal_flip,vertical_flip=flip_image(folder_name+'masked_regions_on_white_bg.png',90)
    
    rotated.save(folder_name+'rotated_image.jpg')
    horizontal_flip.save(folder_name+'horizontally_flipped_image.jpg')
    vertical_flip.save(folder_name+'vertically_flipped_image.jpg')
    
    image_array_flipped, detections_flipped = grounded_segmentation(
        image=folder_name+'horizontally_flipped_image.jpg',
        labels=[labels],
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    
    bbox_flipped=detections_flipped[0].box
    print(bbox_flipped)
    
    cropped_flipped_image=crop_image(folder_name+'horizontally_flipped_image.jpg',(bbox_flipped.xmin,bbox_flipped.ymin,bbox_flipped.xmax,bbox_flipped.ymax))
    cropped_flipped_image.save(folder_name+'cropped_horizontally_flipped.jpg')
    
    final_output=image_overlay(folder_name+'clean_background.jpg',folder_name+'cropped_horizontally_flipped.jpg',(bbox.xmin,bbox.ymin))
    final_output.save(output_path)
    print("Image saved at output path")
if __name__ == '__main__':
    main()



import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from utility_functions import *
import argparse
from PIL import Image
import os 
torch.cuda.empty_cache()
from pathlib import Path


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

    plot_detections_mask(image_array, detections,save_name=output_path)
    print("Image saved in output path")
    
    # print(bbox)

if __name__ == '__main__':
    main()



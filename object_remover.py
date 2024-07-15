
import numpy as np
import pandas as pd

import os
from datetime import datetime
from PIL import Image
import pickle
from io import BytesIO
from copy import deepcopy

from src.core import process_inpaint
import cv2

def clean_background(input_path,mask_path,output_path):

    img_input = Image.open(input_path)
    im = Image.open(mask_path).convert("RGBA")
    im=np.array(im)
    # print(im)
    for i in range(len(im)):
        for j in range(len(im[i])):
            if np.array_equal(im[i][j], [255, 255, 255, 255]):
                im[i][j] = np.array([0, 0, 0, 0])  
        
    # with open('/home/vaibhavsharma/Desktop/rfp/unstructured_excels/object_remover/remove-photo-object/pickle_files/main_image.pkl', 'wb') as file:
    #     # Use pickle to dump the data into the file
    #     pickle.dump(np.array(img_input), file)
        
    # with open('/home/vaibhavsharma/Desktop/rfp/unstructured_excels/object_remover/remove-photo-object/pickle_files/streamlit_mask.pkl', 'rb') as file:
    #     # Use pickle to dump the data into the file
    #     im=pickle.load(file)
    output = process_inpaint(np.array(img_input), np.array(im)) #TODO Put button here
    img_output = Image.fromarray(output).convert("RGB")
    
    img_output.save(output_path)





This project focuses on two main tasks: zero-shot image segmentation and generating new camera views of objects. Follow the instructions below to set up and run the tasks.

This repository also consists of a report to look at both the process flow charts and some results and discussion

Installation Steps


```bash
conda create -n new_env python=3.9
conda activate new_env
cd zero-shot-segmentation
pip install --upgrade -q git+https://github.com/huggingface/transformers
```

Next, install the additional required packages:

```bash
pip install -r requirements.txt
```



### Task 1: Zero-Shot Image Segmentation

To perform zero-shot image segmentation, use the following command:

```bash
python run1.py --image INPUT_IMG_PATH --object OBJECTNAME(example: Lamp) --output OUTPUT_IMG_PATH 
```

This script will segment the specified object (`lamp`) in the given image and save the output to the specified location.

### Task 2: Generate New Camera View of Object

Before running task two make sure this model is downloaded and saved in the assets folder [Model](https://drive.google.com/file/d/13iMRwZP8tqNcKispSxsepqIP7D5Z0w3l/view?usp=sharing)
 the model is required to remove object from background image , and obtain a background image without the object
To generate a new camera view of an object, run the following command:

```bash
python run2.py --image INPUT_IMG_PATH --object OBJECTNAME(example: Lamp) --output OUTPUT_IMG_PATH 
```

This script processes the image to create a new view of the `lamp` and saves the result to the designated output directory. It will also create a folder with the object name and it will have intermediate files , including the background image without the object , all these intermediate files are necessary in obtaining the output




# AI701: Robustness Evaluation Of Textual Inversion Model and Multi-Concept Inference

This repository contains <fill>:

1. **SD Textual Inversion Training** - A notebook for training textual inversion models using state-of-the-art tools.
2. **Multi-Concept Inference** - A notebook focused on inference tasks involving multiple concepts in AI models.
3. **Evaluataion Metrices** - This repository includes tools for evaluating AI-generated outputs using the following metrics:
     **FID Score**: Computes the Fréchet Inception Distance to measure similarity between real and generated images.
     **CLIP Score**: Evaluates alignment between images and predefined textual prompts using CLIP models.
     **Concept Accuracy Score**: Assesses adherence to specific concepts by combining ResNet predictions with CLIP-based similarity.
4. **Preprocessing Script** - A script for preparing and processing the training data for the model.

All notebooks leverage cutting-edge frameworks such as PyTorch, HuggingFace, and more, providing robust tools for researchers and developers.

---

## Notebooks Overview

### 1. SD Textual Inversion Training
This notebook facilitates:
- Training textual inversion models.
- Utilising tools like `diffusers` and `transformers` for model development.
- Accelerated computation with `torch` and `accelerate`.

### 2. Multi-Concept Inference
This notebook enables:
- Inference tasks involving multi-concept datasets.
- Utilising pre-trained models with `diffusers` and `transformers`.
- Efficient computation with advanced libraries like `torch`.

### 3. <Ishita to fill>


### 4. Preprocessing Script
This script handles data preprocessing for training. It prepares and formats raw data, applies transformations, and saves the processed data for the model training phase. The script applies various augmentations to the training dataset:

- **Gaussian Noise**: Adds noise with adjustable severity (low, medium, high).
- **Motion Blur**: Applies motion blur with varying kernel sizes.
- **Color Jitter**: Randomly adjusts brightness, contrast, saturation, and hue of the image.
- **Random Occlusion**: Randomly occludes portions of the image to simulate real-world occlusion scenarios.
  


---

## Requirements

The following libraries are required to run both notebooks:

- `IPython`
- `PIL`
- `accelerate`
- `argparse`
- `diffusers`
- `huggingface_hub`
- `itertools`
- `math`
- `numpy`
- `os`
- `random`
- `subprocess`
- `time`
- `torch`
- `torchvision`
- `tqdm`
- `transformers`
- `opencv-python`


---

## Installation

### Using pip
Install the required libraries directly:

```bash
pip install IPython pillow accelerate argparse diffusers huggingface_hub numpy torch torchvision tqdm transformers opencv-python
```

## Using a requirements.txt file
Create a requirements.txt file with the following content:

IPython
pillow
accelerate
argparse
diffusers
huggingface_hub
numpy
torch
torchvision
tqdm
transformers
opencv-python

Then run:
```
pip install -r requirements.txt
```

## How to Run the Notebooks

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd <repository-name>
   ```
2. Ensure you have all dependencies installed (as mentioned above).
3. Launch the notebooks:
  3.1. For SD Textual Inversion Training:
   ```
   jupyter notebook sd_textual_inversion_training_1_latest.ipynb
   ```
  3.2. For Multi-Concept Inference:
  ```
  jupyter notebook multi_concept_inference.ipynb
  ```

## Training Dataset
The dataset used for training consists of images related to various **UAE-MBZUI themes**, categorized into the following groups:

- **MBZUAI-related Objects**: 5 objects (cup, logo, mainbuilding, papercup, bag)
- **UAE Landmarks**: 3 objects (Burj Khalifa, Sheikh Zayed Grand Mosque, Emirates Palace)
- **UAE Animals**: 4 objects (Falcon, Camel, Horse, Oryx)
- **UAE Cuisine**: 3 objects (Harees, Balaleet, Luqimat)

You can access the dataset from the following link: [Training Dataset](https://mbzuaiac-my.sharepoint.com/personal/anjali_khantaal_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fanjali%5Fkhantaal%5Fmbzuai%5Fac%5Fae%2FDocuments%2FProjects%2FTraining%5FData)

## How to Run the Preprocessing Script

Follow these steps to run the preprocessing script:

1. **Place Your Training Images**  
   Place your training images in a folder named `Training_Dataset` inside the root directory of the repository.

2. **Run the Preprocessing Script**  
   To run the preprocessing script, execute the following command in your terminal:

   ```bash
   python preprocessing.py
   
This script will:

- Process the images in the `Training_Dataset` folder.
- Apply augmentations with three different severity levels: low, medium, and high.
- Save the preprocessed images in a new folder called `Preprocessed`.

This project is inspired by the Textual-inversion model present on HuggingFace: <enter link>



link to access output: https://mbzuaiac-my.sharepoint.com/:f:/r/personal/anjali_khantaal_mbzuai_ac_ae/Documents/Projects/Training_Outputs?csf=1&web=1&e=pabbcp

link to report:

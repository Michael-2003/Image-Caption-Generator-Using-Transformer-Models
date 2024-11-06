![image](https://github.com/user-attachments/assets/3af2cc81-1a27-42af-a7a2-f1563d4ce187)# **Image Caption Generator Using Transformer Models**

## **Project Overview**

This project aims to develop an Image Caption Generator using deep learning techniques. The model utilizes a **pre-trained EfficientNet** for feature extraction and a **Transformer-based encoder-decoder architecture** for generating captions. The system takes an image as input and generates a natural language description of its content.

### **Features:**
- **Image Captioning**: Automatically generates captions for input images.
- **Transformer Model**: Uses Transformer architecture for generating coherent and contextually relevant captions.
- **EfficientNet**: A pre-trained EfficientNet model is used to extract features from images.
- **GUI**: A simple graphical interface that allows users to upload images and get captions.
- **Jupyter Notebook**: Includes a notebook with all code for training, evaluation, and inference.

---

## **File Structure**

- **`parameters.py`**  
  Contains configuration settings for the model, such as dataset paths, image size, batch size, sequence length, embedding dimensions, and training epochs.

- **`model.py`**  
  Defines the architecture of the image captioning model, including the EfficientNet-based CNN for feature extraction and the Transformer-based encoder-decoder structure. It also includes the forward pass of the model and loss functions.

- **`utility.py`**  
  Provides helper functions for preprocessing, including image resizing, caption tokenization, vocabulary management, text vectorization, and mask creation.

- **`generate_caption.py`**  
  Implements the logic for loading an image, processing it, and generating a caption using the trained model. This script is used for real-time caption generation.

- **`gui.py`**  
  Implements a simple graphical user interface (GUI) using Tkinter for users to upload images and receive captions.

- **`ImageCaptioning.ipynb`**  
  A Jupyter notebook that contains all the code for training the model, generating captions, and performing evaluations. This notebook is useful for anyone who wants to run the code interactively in a notebook environment.

---

## **Dataset**

The model is trained on the **Flickr8k dataset**, which contains 8,000 images, each paired with multiple descriptive captions. This dataset is ideal for training image captioning models. You can access and download the dataset from Kaggle using the link below:

[Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## **How to Use**

### **1. Clone the repository**:
    ```bash
    git clone https://github.com/your-username/image-caption-generator.git
    cd image-caption-generator
    ```

### **2. Install dependencies**:
   The project requires several libraries for deep learning, image processing, and GUI development. You can install them via `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### **3. Training the Model**:
   - Set up the dataset path and model parameters in `parameters.py`.
   - Run the `train.py` script to train the model on the Flickr8k dataset:
    ```bash
    python train.py
    ```

### **4. Generate Captions**:
   After training, run the `generate_caption.py` script to input an image and receive a generated caption:
    ```bash
    python generate_caption.py --image_path /path/to/image.jpg
    ```

---

## **Model Architecture**

The architecture of the model combines two powerful components:

### **EfficientNet CNN**

The EfficientNet model is used to extract image features from the input image. The extracted features are passed to the Transformer decoder for caption generation.

### **Transformer**

The Transformer architecture, consisting of an encoder-decoder structure, is used to generate captions. The encoder processes the image features, and the decoder generates the sequence of words forming the caption.

![Model Architecture](path_to_architecture_image)  
*(Replace "path_to_architecture_image" with the actual image file path)*

---

## **GUI Interface**

The user-friendly interface allows users to easily upload images and view the generated captions. Here is a screenshot of the GUI:

![GUI Screenshot](![Uploading image.png…])
  
*(Replace "path_to_gui_image" with the actual image file path)*

---

## **Training the Model**

### **Training Setup**

- Set up the dataset path and model parameters in `parameters.py`.
- Train the model using your dataset by executing `train.py`.

```bash
python train.py

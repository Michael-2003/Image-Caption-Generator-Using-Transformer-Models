# **Image Caption Generator Using Transformer Models**

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

## **GUI Interface**

The **GUI** provides a seamless and user-friendly way for users to interact with the image captioning model. With the interface, users can easily upload images and instantly view the generated captions. This provides a convenient way to test and explore the model's performance without needing to interact with the command line.

### **Features of the GUI:**
- **Image Upload**: Upload images in formats such as PNG or JPEG.
- **Caption Generation**: Automatically generates captions for the uploaded images.
- **Real-time Results**: Display of the generated caption immediately after the image is processed.
- **User-friendly Layout**: Simple and intuitive interface, making it accessible for users with no programming experience.

### **GUI Screenshot**  
Here’s what the GUI looks like when opened:

![GUI Interface.png](https://github.com/Michael-2003/Image-Caption-Generator-Using-Transformer-Models/blob/328517e87f33bd71d338d5b2fe7d537219fdf8d3/GUI%20Interface.png)

### **Test Example**  
Here’s an example of the GUI in action, showing an uploaded image and the generated caption:

![GUI Interface_test_example.png](https://github.com/Michael-2003/Image-Caption-Generator-Using-Transformer-Models/blob/c7421c92005bc7f3d5dd5fed2bcde361aeb9136d/GUI%20interface_test_example.png)

### **How to Use the GUI:**
1. **Launch the GUI**: Run the following command in your terminal:
    ```bash
    python gui.py
    ```
   This will open the GUI window.

2. **Upload an Image**: Click on the "Upload Image" button to select an image from your computer. Supported formats include JPEG and PNG.

3. **View the Caption**: After uploading the image, the model will generate a caption that will be displayed below the image in the GUI.

4. **Test Multiple Images**: Try uploading different images to see how well the model generates captions across a variety of inputs.

---
## **Saving Model Weights and Vocabulary**

During training, the model weights and vocabulary are saved for future use. This allows for easy loading of the trained model and its vocabulary during inference.

### **Saving Model Weights**:
The model weights are saved in a `.h5` file after training. This is done through the **Jupyter Notebook** and not any Python file:

```python
caption_model.save_weights('ICM_weights.h5')
```
### **Saving Vocabulary**:
The vocabulary is adapted and saved as a `vocab.pkl` file in the notebook as well. This allows the model to reference the same vocabulary during inference:

```python
def adapt_vectorization_layer(captions):
    # Adapt the vectorization layer to the captions
    vectorization.adapt(captions)

    # Save the adapted vocabulary
    vocab = vectorization.get_vocabulary()
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

adapt_vectorization_layer(text_data)

```
These steps are executed within the `ImageCaptioning.ipynb` notebook, ensuring the trained model and its vocabulary are properly saved and can be loaded for inference later.
---
## **Model Architecture**

The architecture of the model combines two powerful components:

### **EfficientNet CNN**

The EfficientNet model is used to extract image features from the input image. The extracted features are passed to the Transformer decoder for caption generation.

### **Transformer**

The Transformer architecture, consisting of an encoder-decoder structure, is used to generate captions. The encoder processes the image features, and the decoder generates the sequence of words forming the caption.

![Model Architecture](https://github.com/Michael-2003/Image-Caption-Generator-Using-Transformer-Models/blob/328517e87f33bd71d338d5b2fe7d537219fdf8d3/Model%20Architecture.jpg)  


---



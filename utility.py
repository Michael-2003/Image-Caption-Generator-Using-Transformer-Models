from model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
# from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import TextVectorization
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm_notebook
from collections import Counter
import pickle

def custom_standardization(input_string):
    # Lowercasing all of the captions
    lowercase = tf.strings.lower(input_string)
    # Charecters to remove
    strip_chars = "!\"#$%&'()*+,-./:;=?@[]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)



# Re-create the TextVectorization layer with the loaded vocabulary
vectorization = TextVectorization(
    max_tokens=len(vocab),
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization
)

vectorization.set_vocabulary(vocab)


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img



def get_inference_model():

  cnn_model = get_cnn_model()
  encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=2)
  
  decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=3, vocab_size=VOCAB_SIZE)
  
  caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)

  # It's necessary to initialize the model (call it once)
  cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  training = False
  decoder_input = tf.keras.layers.Input(shape=(None,))
  
  caption_model(inputs=[cnn_input, tf.constant(False, dtype=tf.bool), decoder_input])


  return caption_model

model = get_inference_model()
model.load_weights("ICM_weights.h5")

vocab = vectorization.get_vocabulary()
INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}
MAX_DECODED_SENTENCE_LENGTH = SEQ_LENGTH - 1





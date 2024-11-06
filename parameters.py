import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')


IMAGES_PATH = "/kaggle/input/flickr8k/Images"

CAPTIONS_PATH = "/kaggle/input/flickr8k/captions.txt"

IMAGE_SIZE = (299, 299)

SEQ_LENGTH = 25

VOCAB_SIZE = 10000

EMBED_DIM = 512

FF_DIM = 512

BATCH_SIZE = 512

EPOCHS = 30
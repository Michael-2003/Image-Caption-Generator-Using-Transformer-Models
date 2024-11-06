from utility import *
from model import *
from parameters import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')




def generate_caption(image_path):
    # Read the image from the disk
    image = decode_and_resize(image_path)

    # Expand dimensions to match model input shape
    image = tf.expand_dims(image, 0)

    # Pass the image through the CNN to get features
    image_features = model.cnn_model(image)

    # Encode the image features
    encoded_img = model.encoder(image_features, training=False)

    # Initialize the caption with the start token
    decoded_caption = "<start> "
    
    # Generate the caption token by token
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = INDEX_TO_WORD[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    # Clean up the generated caption
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    
    return decoded_caption


def get_attention_weights(image_path):
    # Read the image from the disk
    image = decode_and_resize(image_path)

    # Expand dimensions to match model input shape
    image = tf.expand_dims(image, 0)

    # Pass the image through the CNN to get features
    image_features = model.cnn_model(image)

    # Encode the image features
    encoded_img = model.encoder(image_features, training=False)

    # Generate the caption to get attention weights
    decoded_caption = "<start> "
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        _, attention_weights = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask, return_attention=True)
        predictions, _ = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, -1, :])
        sampled_token = INDEX_TO_WORD[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    return attention_weights


def plot_attention(image, result, attention_plot):
    temp_image = np.array(image)

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result.split())

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2 + 1, len_result // 2, l + 1)
        ax.set_title(result.split()[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

caption = generate_caption("WhatsApp Image 2024-05-28 at 02.27.04_6f76b3b0.jpg")
print("Generated Caption:", caption)

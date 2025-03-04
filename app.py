# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the Keras model
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("keras_model.h5")  # Ensure correct filename
#     return model

# # Load class labels
# @st.cache_resource
# def load_labels():
#     with open("labels.txt", "r") as f:
#         labels = [line.strip() for line in f.readlines()]
#     return labels

# # Load model and labels
# model = load_model()
# labels = load_labels()

# # Streamlit UI
# st.title("Teachable Machine Model in Streamlit")
# st.write("Upload an image to classify:")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess Image
#     img = image.resize((224, 224))  # Adjust size based on your model
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make Prediction
#     prediction = model.predict(img_array)
    
#     # Get Predicted Class
#     class_index = np.argmax(prediction)
#     confidence = np.max(prediction) * 100  # Confidence percentage
    
#     # Display Prediction
#     st.write(f"Prediction: {labels[class_index]}")  # Map index to label
#     st.write(f"Confidence: {confidence:.2f}%")




from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

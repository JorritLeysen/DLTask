import streamlit as st
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

NUM_CLASSES = 6
IMG_SIZE = 32
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

@st.cache_data
def load_images():
    return './images'
    
@st.cache_resource
def load_pretrained_model():
    # Load the pre-trained model
    model = tf.keras.models.load_model("./saved_models/meansoftransport.tf")
    return model

def main():
    st.title("Image Classifier")

    # Log memory usage
    process = psutil.Process(os.getpid())
    st.write(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Model Training Settings
    epochs = st.slider("Number of epochs:", min_value=1, max_value=50, value=20)

    if st.button("Train Model"):
        # Execute model training
        st.text("Training in progress...")

        # Create progress bar for training
        progress_bar = st.progress(0)

        # Create the training dataset from the 'images' directory
        train_ds = image_dataset_from_directory(
            directory=load_images(),
            labels='inferred',
            label_mode='categorical',
            batch_size=4,
            image_size=(IMG_SIZE, IMG_SIZE),
            validation_split=0.2,
            subset='training',
            seed=123
        )

        # Create the validation dataset from the 'images' directory
        validation_ds = image_dataset_from_directory(
            directory=load_images(),
            labels='inferred',
            label_mode='categorical',
            batch_size=4,
            image_size=(IMG_SIZE, IMG_SIZE),
            validation_split=0.2,
            subset='validation',
            seed=123
        )

        # Train the model and get the trained model
        trained_model = train_model(train_ds, validation_ds, epochs, progress_bar)

        # Evaluate the model on the test dataset
        test_ds = image_dataset_from_directory(
            directory='./testimages',
            labels='inferred',
            label_mode='categorical',
            batch_size=4,
            image_size=(IMG_SIZE, IMG_SIZE)
        )
        test_loss, test_acc = trained_model.evaluate(test_ds)
        st.subheader("Test Accuracy")
        st.write(f'Test accuracy: {test_acc:.4f}')
    else:
        # Load the pre-trained model
        pretrained_model = load_pretrained_model()

        # Use the pre-trained model for predictions
        st.subheader("Make Predictions")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            # Preprocess the image for prediction
            img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array /= 255.0  # Normalize to [0, 1]

            # Make predictions
            predictions = pretrained_model.predict(img_array)

            # Display predictions
            st.write("Predictions:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"Class {i}: Probability {prob:.4f}")

if __name__ == "__main__":
    main()

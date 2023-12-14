import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 6
IMG_SIZE = 128

@st.cache_resource
def load_pretrained_model():
    # Load the pre-trained model
    model = load_model("saved_models/meansoftransport.tf")
    return model

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Image Classifier")

    class_names = ['bicycle', 'boat', 'car', 'motorcycle', 'plane', 'truck']

    # Load the pre-trained model
    pretrained_model = load_pretrained_model()

    # Use the pre-trained model for predictions
    st.subheader("Make Predictions")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Preprocess the image for prediction
        img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array /= 255.0  # Normalize to [0, 1]

        # Make predictions
        predictions = pretrained_model.predict(img_array)

        # Display predictions
        st.write("Predictions:")
        for i, prob in enumerate(predictions[0]):
            class_name = class_names[i]
            st.write(f"Class {class_name}: Probability {prob:.4f}")

    # Evaluate the model on the test dataset
    st.subheader("Test on testimages")

    # Create the testing dataset from the 'testimages' directory
    test_images_folder = './testimages'
    test_ds = image_dataset_from_directory(
        directory=test_images_folder,
        labels='inferred',
        label_mode='categorical',
        batch_size=4,
        image_size=(IMG_SIZE, IMG_SIZE)
    )

    # Store true and predicted labels during testing
    y_true = []
    y_pred = []

    for x, y in test_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(pretrained_model.predict(x), axis=1))

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(NUM_CLASSES)], yticklabels=[f'Class {i}' for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Display test accuracy
    test_loss, test_acc = pretrained_model.evaluate(test_ds)
    st.subheader("Test Accuracy")
    st.write(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()

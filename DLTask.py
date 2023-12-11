import streamlit as st
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
import requests
from selenium.webdriver.common.by import By
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory

NUM_CLASSES = 6
IMG_SIZE = 64
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

def create_model():
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")  # Use softmax for multi-class classification
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_ds, validation_ds, epochs, progress_bar):
    model = create_model()
    num_batches = len(train_ds)
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_ds):
            # Training steps go here
            history = model.fit(train_ds,
                validation_data = validation_ds,
                epochs = epochs
                )
            
            # Update the progress bar
            progress_bar.progress((i + 1) / num_batches)

        # Validation steps go here
        # validation_loss, validation_accuracy = model.evaluate(validation_ds)

        # Display training progress for each epoch
        st.write(f"Epoch {epoch + 1}/{epochs} - Training Loss: ... - Training Accuracy: ... - Validation Loss: ... - Validation Accuracy: ...")

    st.success("Training complete!")

def main():
    st.title("Google Images Scraper & Classifier with Streamlit")

    # Sidebar for Model Training
    st.sidebar.header("Model Training Settings")
    epochs = st.sidebar.slider("Number of epochs:", min_value=1, max_value=50, value=20)

    if st.sidebar.button("Train Model"):
        # Execute model training
        st.sidebar.text("Training in progress...")

        # Create progress bar for training
        progress_bar = st.sidebar.progress(0)

        # Create the training dataset from the 'images' directory
        train_ds = image_dataset_from_directory(
            directory='./images',
            labels='inferred',
            label_mode='categorical',
            batch_size=16,
            image_size=(IMG_SIZE, IMG_SIZE),
            validation_split=0.2,
            subset='training',
            seed=123
        )

        # Create the validation dataset from the 'images' directory
        validation_ds = image_dataset_from_directory(
            directory='./images',
            labels='inferred',
            label_mode='categorical',
            batch_size=16,
            image_size=(IMG_SIZE, IMG_SIZE),
            validation_split=0.2,
            subset='validation',
            seed=123
        )

        # Create the testing dataset from the 'testimages' directory
        test_ds = image_dataset_from_directory(
            directory='./testimages',
            labels='inferred',
            label_mode='categorical',
            batch_size=16,
            image_size=(IMG_SIZE, IMG_SIZE)
        )

        train_model(train_ds, validation_ds, epochs, progress_bar)

        # Evaluate the model on the test dataset
        test_loss, test_acc = model.evaluate(test_ds)
        st.subheader("Test Accuracy")
        st.write(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()

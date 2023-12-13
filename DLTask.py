import streamlit as st
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

    # Lists to store training history
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # Lists to store true and predicted labels
    y_true = []
    y_pred = []

    for epoch in range(epochs):
        # Training steps
        history = model.fit(train_ds, validation_data=validation_ds, steps_per_epoch=1, epochs=1)

        # Update training history lists
        training_loss.append(history.history['loss'][0])
        training_accuracy.append(history.history['accuracy'][0])
        validation_loss.append(history.history['val_loss'][0])
        validation_accuracy.append(history.history['val_accuracy'][0])

        # Update the progress bar
        progress_bar.progress((epoch + 1) / epochs)

        # Display training progress for each epoch
        st.write(f"Epoch {epoch + 1}/{epochs} - "
                 f"Training Loss: {training_loss[-1]:.4f} - "
                 f"Training Accuracy: {training_accuracy[-1]:.4f} - "
                 f"Validation Loss: {validation_loss[-1]:.4f} - "
                 f"Validation Accuracy: {validation_accuracy[-1]:.4f}")

        # Store true and predicted labels during training
        for x, y in validation_ds:
            y_true.extend(np.argmax(y.numpy(), axis=1))
            y_pred.extend(np.argmax(model.predict(x), axis=1))

    st.success("Training complete!")

    # Visualize training history
    st.header("Training History")

    # Training Loss
    st.subheader("Training Loss")
    st.line_chart(training_loss, use_container_width=True)

    # Training Accuracy
    st.subheader("Training Accuracy")
    st.line_chart(training_accuracy, use_container_width=True)

    # Validation Loss
    st.subheader("Validation Loss")
    st.line_chart(validation_loss, use_container_width=True)

    # Validation Accuracy
    st.subheader("Validation Accuracy")
    st.line_chart(validation_accuracy, use_container_width=True)

    # Confusion Matrix
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(NUM_CLASSES)], yticklabels=[f'Class {i}' for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()
    
    # Return the trained model
    return model

def main():
    st.title("Google Images Scraper & Classifier with Streamlit")

    # Model Training Settings
    epochs = st.slider("Number of epochs:", min_value=1, max_value=50, value=20)

    if st.button("Train Model"):
        # Execute model training
        st.text("Training in progress...")

        # Create progress bar for training
        progress_bar = st.progress(0)

        # Create the training dataset from the 'images' directory
        train_ds = image_dataset_from_directory(
            directory='./images',
            labels='inferred',
            label_mode='categorical',
            batch_size=8,
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
            batch_size=8,
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
            batch_size=8,
            image_size=(IMG_SIZE, IMG_SIZE)
        )

        # Train the model and get the trained model
        trained_model = train_model(train_ds, validation_ds, epochs, progress_bar)

        # Evaluate the model on the test dataset
        test_loss, test_acc = trained_model.evaluate(test_ds)
        st.subheader("Test Accuracy")
        st.write(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()

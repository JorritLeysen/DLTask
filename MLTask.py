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

def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = 1):

    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)
    
    # Build the Google Images URL
    search_url = "https://www.google.com/search?q={}&tbm=isch".format(query)
    wd.get(search_url)

    id_value = "yDmH0d"
    xpath_expression = f"//*[@id='{id_value}']/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[1]/div/div/button"
    button_element = wd.find_element(By.XPATH, xpath_expression)

    # Click the button
    button_element.click()

    image_urls = set()
    image_count = 0

    while image_count < max_links_to_fetch:
        scroll_to_end(wd)
        time.sleep(sleep_between_interactions)

        # Extract image URLs directly from the page source using regular expressions
        page_source = wd.page_source
        matches = re.findall(r'img\ssrc=\"(https:[^"]+)', page_source)
        image_urls.update(matches)

        image_count = len(image_urls)

        if image_count >= max_links_to_fetch:
            print(f"Found {image_count} image links")
            break
        else:
            print(f"Found {image_count} image links, looking for more ...")
            time.sleep(sleep_between_interactions)  # Adjust as needed

    return image_urls

def download_images(folder_path: str, file_prefix: str, image_urls: set):
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            # Ensure that the image is valid
            if response.status_code == 200:
                with open(os.path.join(folder_path, f"{file_prefix}_{i}.jpg"), 'wb') as file:
                    file.write(response.content)
        except Exception as e:
            print(f"Error downloading image {i + 1}: {e}")

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

def main():
    st.title("Google Images Scraper & Classifier with Streamlit")

    # Sidebar for Scraping
    st.sidebar.header("Image Scraping Settings")
    query = st.sidebar.text_input("Enter search query:")
    max_links_to_fetch = st.sidebar.number_input("Max links to fetch:", value=150)

    if st.sidebar.button("Scrape Images"):
        # Execute the scraping and downloading
        st.sidebar.text("Scraping in progress...")

        # Disable webdriver logs to avoid cluttering the Streamlit interface
        webdriver_options = webdriver.ChromeOptions()
        webdriver_options.add_argument("--headless")
        webdriver_options.add_argument("--disable-gpu")
        webdriver_options.add_argument("--log-level=3")
        wd = webdriver.Chrome(options=webdriver_options)

        try:
            image_urls = fetch_image_urls(query, max_links_to_fetch)
            folder_path = f"./images/{query.lower()}"
            os.makedirs(folder_path, exist_ok=True)
            download_images(folder_path, query.lower(), image_urls)
            st.sidebar.text(f"Scraping complete. Images saved in {folder_path}")
        except Exception as e:
            st.sidebar.text(f"Error: {e}")
        finally:
            wd.quit()

    # Sidebar for Model Training
    st.sidebar.header("Model Training Settings")

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

    # Train the model
    model = create_model()
    history = model.fit(train_ds, validation_data=validation_ds, epochs=20)

    # Display the training history
    st.subheader("Training History")
    st.line_chart(history.history)

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_ds)
    st.subheader("Test Accuracy")
    st.write(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()

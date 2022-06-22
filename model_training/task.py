import os

import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from model import get_model, get_callbacks

PATH_TO_DATA = Path('./data')
PATH_TO_SAVE_MODEL = Path('./best_model')

NUM_EPOCHS = 50
IMAGE_SHAPE = [400, 400]


def main():
    all_raw_images, all_raw_labels, categories = load_all_raw_images()
    print("num categories: ", len(categories))
    all_preprocessed_images, all_preprocessed_labels = preprocess_images_and_labels(all_raw_images, all_raw_labels,
                                                                                    categories)
    x_train, x_test, y_train, y_test = train_test_split(all_preprocessed_images, all_preprocessed_labels, test_size=0.4)
    model = get_model(IMAGE_SHAPE, len(categories))
    model.summary()
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS,
                        validation_data=(x_test, y_test), callbacks=get_callbacks(PATH_TO_SAVE_MODEL))
    evaluate_model(history, model, x_test, y_test)


def load_all_raw_images():
    print("loading data...", end=" ")
    all_raw_images = []
    all_raw_labels = []
    categories = []
    for path_to_directory in PATH_TO_DATA.iterdir():
        label = path_to_directory.stem
        categories.append(label)
        for path_to_image in path_to_directory.iterdir():
            raw_img = cv2.imread(str(path_to_image), cv2.IMREAD_COLOR)
            all_raw_images.append(raw_img)
            all_raw_labels.append(label)
    print("data loaded.")
    return all_raw_images, all_raw_labels, categories


def preprocess_images_and_labels(all_raw_images, all_raw_labels, categories):
    print("preprocessing images...", end=" ")
    preprocessed_images = []
    preprocessed_labels = []
    for raw_img, raw_label in zip(all_raw_images, all_raw_labels):
        resized_img = cv2.resize(raw_img, IMAGE_SHAPE)
        categorical_label = categories.index(raw_label)
        preprocessed_images.append(resized_img)
        preprocessed_labels.append(categorical_label)
    print("done.")
    return np.array(preprocessed_images), np.array(preprocessed_labels)


def evaluate_model(history, model, x_test, y_test):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    main()

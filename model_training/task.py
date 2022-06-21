import cv2
from pathlib import Path

PATH_TO_DATA = Path('./data')

# load and preprocess all images from the data bank

def main():
    all_raw_images = load_all_raw_images()


def load_all_raw_images():
    for path_to_image in PATH_TO_DATA.iterdir():
        myimg = cv2.imread(str(path_to_image), cv2.IMREAD_COLOR)
        myimg = cv2.resize(myimg, [240, 240])
        cv2.imshow('Display_image', myimg)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()

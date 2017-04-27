import numpy as np
import cv2
import glob
import matplotlib.image as mpimg



def smart_img(img):
    if np.max(img) <= 1:
        # Just return it if we're already scaled correctly.
        return img
    else:
        # If it's in 8-bit format, normalize to 0-1.
        return img.astype(np.float32)/255



def resize_captured_images():
    newcars = glob.glob('./images/vid extracts/originals/*')

    counter = 1

    for img in newcars:
        # Read in image
        image = mpimg.imread(img)

        # Smart convert to floating point
        image = smart_img(image)

        # Resize it
        resized = cv2.resize(image, (64, 64))

        mpimg.imsave('./images/vid extracts/{}.png'.format(counter), resized)
        counter += 1

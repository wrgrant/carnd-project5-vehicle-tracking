import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import myplot
import lesson_functions




def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes




def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img




heat = None


def do_it(box_list, image):
    global heat
    # Going to keep 'heat' as a module variable so I can keep it in scope across multiple
    # calls to this function.
    if heat is None:
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Threshold below which contribution will not be counted for boundary
    # box drawing.
    label_threshold = 4
    # Threshold below which pixels will be set to 0
    heat_threshold = 2

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat_thresholded = np.copy(heat)
    heat_thresholded[heat < label_threshold] = 0

    # Find final boxes from heatmap using label function
    labels = label(np.clip(heat_thresholded, 0, 255))
    labeled_boxes_img = draw_labeled_bboxes(np.copy(image), labels)
    raw_boxes_img = lesson_functions.draw_boxes(np.copy(image), box_list)


    if False:
        myplot.plot_triple(raw_boxes_img, heat, labeled_boxes_img, 'Incoming hot windows', 'Heatmap', 'Result boundaries', cmap2='hot')


    # Decay the values in heat a bit.
    heat *= 0.6
    # And ensure no values go below 0.
    heat = np.clip(heat, 0, 255)
    # Kill off any residuals.
    heat[heat < heat_threshold] = 0

    return labeled_boxes_img

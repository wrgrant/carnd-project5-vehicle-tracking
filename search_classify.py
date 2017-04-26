import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import lesson_functions
import pickle
import detect_multiple
import myplot
import util
import averager
from joblib import delayed



prl_context = None


dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]
spatial_feat = dist_pickle["spatial_feat"]



def search_window(img, window):
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        features = lesson_functions.single_img_features(test_img, color_space, spatial_size, hist_bins,
                                                        orient, pix_per_cell, cell_per_block,
                                                        hog_channel, spatial_feat, hist_feat, hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = X_scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = svc.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        # myplot.plot(test_img, '{}'.format(prediction))
        if prediction == 1:
            return window



# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows):
    # 1) Create an empty list to receive positive detection windows
    # on_windows = []
    # 2) Iterate over all windows in the list
    # for window in windows:
    on_windows = prl_context(delayed(search_window) (img, window) for window in windows)

    # Filter out 'None's from list.
    on_windows = [x for x in on_windows if x is not None]

    # 8) Return windows for positive detections
    return on_windows







def do_it(in_img, prl):
    global prl_context
    prl_context = prl

    # This function automatically ensures the image is scaled correctly.
    scaled_img = util.smart_img(in_img)
    # myplot.plot(image)


    windows1 = lesson_functions.slide_window(scaled_img, x_start_stop=[600, 1280], y_start_stop=[400, 700],
                           xy_window=(150, 200), xy_overlap=(0.8, 0.8))

    windows2 = lesson_functions.slide_window(scaled_img, x_start_stop=[700, 1200], y_start_stop=[400, 550],
                           xy_window=(180, 100), xy_overlap=(0.9, 0.9))

    windows = windows1 + windows2
    # windows = windows1
    # windows = windows2

    hot_windows = search_windows(scaled_img, windows)
    averaged_windows_deque = averager.do_search(hot_windows)







    if False:
        # This bit just plots every window and the resulting positive windows.
        raw_windows = lesson_functions.draw_boxes(scaled_img, windows, color=(0, 0, 1), thick=6)

        # Make a straight list of windows so draw_boxes() can plot it.
        averaged_windows_list = []
        for lst in averaged_windows_deque:
            for item in lst:
                averaged_windows_list.append(item)

        pos_windows = lesson_functions.draw_boxes(scaled_img, averaged_windows_list, color=(0, 0, 1), thick=6)
        myplot.plot_double(raw_windows, pos_windows)


    # Do the heat-mapping stuff.
    # draw_image = np.copy(in_img)
    # draw_image = detect_multiple.do_it(hot_windows, draw_image)
    # plt.imshow(draw_image)
    # plt.show()


    out_img = np.copy(in_img)
    out_img = detect_multiple.do_it(averaged_windows_deque, out_img)

    return out_img



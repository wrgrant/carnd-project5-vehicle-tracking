import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import lesson_functions
import pickle
import detect_multiple
import myplot








# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = lesson_functions.single_img_features(test_img, color_space, spatial_size, hist_bins,
                                                        orient, pix_per_cell, cell_per_block,
                                                        hog_channel, spatial_feat, hist_feat, hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows




def smart_img(img):
    if np.max(img) <= 1:
        # Just return it if we're already scaled correctly.
        return img
    else:
        # If it's in 8-bit format, normalize to 0-1.
        return img.astype(np.float32)/255




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





def do_it(img):
    # This function automatically ensures the image is scaled correctly.
    image = smart_img(img)
    # myplot.plot(image)

    scale = 1

    y_start_stop = [400, 650]
    x_start_stop = [600, 1200]

    windows1 = lesson_functions.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(180, 120), xy_overlap=(0.8, 0.8))

    windows2 = lesson_functions.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    # windows = windows1 + windows2
    windows = windows1
    # windows = windows2

    raw_windows = lesson_functions.draw_boxes(image, windows, color=(0, 0, 1), thick=6)


    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)




    if False:
        # This bit just plots every window and the resulting positive windows.
        pos_windows = lesson_functions.draw_boxes(image, hot_windows, color=(0, 0, 1), thick=6)
        myplot.plot_double(raw_windows, pos_windows)


    # Do the heat-mapping stuff.
    # draw_image = np.copy(img)
    # draw_image = detect_multiple.do_it(hot_windows, draw_image)
    # plt.imshow(draw_image)
    # plt.show()


    return hot_windows



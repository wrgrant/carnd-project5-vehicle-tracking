import glob
import time
from lesson_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.feature import hog
import pickle
# import search_classify
from timer import *
# import pipeline





def train_classifier():
    # images = glob.glob('./images/**/*.png', recursive=True)

    cars = glob.glob('./images/vehicles/KITTI*/*.png')
    newcars = glob.glob('./images/vid extracts/*.png')
    notcars = glob.glob('./images/non-vehicles/**/*.png')

    cars = shuffle(cars)
    notcars = shuffle(notcars)
    # newcars = shuffle(newcars)


    # Use equal car and not car.
    # sample_size = min(len(cars), len(notcars))

    sample_size = 20
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    cars = cars + newcars



    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # HLS good except shadow, YCrCb good both but not as good no shadow

    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    # cell_per_block = 8  # HOG cells per block
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

    spatial_size = (64, 64)  # Spatial binning dimensions
    hist_bins = 128  # Number of histogram bins

    spatial_feat = False  # Spatial features on or off
    hist_feat = False  # Histogram features on or off
    hog_feat = True # HOG features on or off

    tic()
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    toc()
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    toc()
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


    # Create pickle of outputs
    outs = {'svc': svc, 'scaler': X_scaler, 'orient': orient, 'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block, 'spatial_size': spatial_size, 'hist_bins': hist_bins,
            'color_space': color_space, 'hog_channel': hog_channel, 'hist_feat': hist_feat,
            'hog_feat': hog_feat, 'spatial_feat': spatial_feat}

    pickle.dump(outs, open("svc_pickle.p", "wb"))

    # pipeline.do_it(input='project_corrected.mp4', output='./temp_output/project_test.mp4')
    # img = mpimg.imread('test.png')
    # img = img[:, :, 0:3]
    # search_classify.do_it(img)








if __name__ == '__main__':
    train_classifier()
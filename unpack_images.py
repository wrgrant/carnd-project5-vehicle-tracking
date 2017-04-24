import cv2
import csv
import matplotlib.image as mpimg
import myplot
from joblib import Parallel, delayed


rel_path = './images/udacity/object-detection-crowdai/'



def extract_single(row):
    if row[0] == 'xmin':
        return

    xmin = int(row[0])
    ymin = int(row[1])
    xmax = int(row[2])
    ymax = int(row[3])
    file = row[4]
    label = row[5]

    if label == 'Car':
        try:
            img = mpimg.imread(rel_path + file)

            car_img = img[ymin:ymax, xmin:xmax, :]

            # myplot.plot(car_img)

            resized = cv2.resize(car_img, (64, 64))

            file_name = rel_path + '../extracted2/{}-{},{}.png'.format(file, xmin, ymin)
            mpimg.imsave(file_name, resized)
        except:
            print('errored looking in file {}'.format(file))




if __name__ == '__main__':
    with open(rel_path + 'labels.csv') as csvfile:
        reader = csv.reader(csvfile)

        Parallel(n_jobs=8, verbose=1)(delayed(extract_single) (row) for row in reader)

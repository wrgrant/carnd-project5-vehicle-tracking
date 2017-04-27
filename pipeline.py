from moviepy.editor import VideoFileClip
import myplot
import search_classify
import hog_subsample
import pprofile
import matplotlib.image as mpimg
import averager
from joblib import Parallel




prl_context = None




# Debugging function for inspecting specific frames of a video.
def save_frame_at_time(time):
    clip = VideoFileClip('project_corrected.mp4')
    frame = clip.get_frame(time)
    mpimg.imsave('test.png', frame)




def process_images(in_img):
    img = search_classify.do_it(in_img, prl_context)
    return img




def do_it(input, output):
    clip = VideoFileClip(input).subclip(t_start=0)
    # clip = clip.set_duration(20)
    clip = clip.fl_image(process_images)
    clip.write_videofile(output, progress_bar=True, audio=False)




if __name__ == '__main__':
    # save_frame_at_time(35)
    # prof = pprofile.Profile()
    # with prof():
    with Parallel(n_jobs=1, backend='multiprocessing') as prl_context:
        do_it(input='project_corrected.mp4', output='./temp_output/project_test.mp4')
    # prof.callgrind(open('latest.out', 'w'))
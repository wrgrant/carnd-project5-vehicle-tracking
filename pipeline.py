from moviepy.editor import VideoFileClip
import myplot
import search_classify
import hog_subsample
import pprofile
import matplotlib.image as mpimg
import averager




# Debugging function for inspecting specific frames of a video.
def save_frame_at_time(time):
    clip = VideoFileClip('project_corrected.mp4')
    frame = clip.get_frame(time)
    mpimg.imsave('test.png', frame)




def process_images(in_img):
    rectangles = search_classify.do_it(in_img)
    # rectangles = hog_subsample.do_it(in_img)
    img = averager.do_it(in_img, rectangles)
    # myplot.plot(img)

    return img




def do_it(input, output):
    clip = VideoFileClip(input).subclip(t_start=35)
    # clip = VideoFileClip(input).subclip(t_start=7)
    clip = clip.set_duration(10)
    clip = clip.fl_image(process_images)
    clip.write_videofile(output, progress_bar=True, audio=False)





# save_frame_at_time(35)
# prof = pprofile.Profile()
# with prof():
do_it(input='project_corrected.mp4', output='./temp_output/project_test.mp4')
# prof.callgrind(open('latest.out', 'w'))
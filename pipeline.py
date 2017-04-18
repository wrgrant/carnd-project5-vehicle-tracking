from moviepy.editor import VideoFileClip
import myplot
import search_classify
import hog_subsample







def process_images(in_img):
    # img = search_classify.do_it(in_img)
    img = hog_subsample.do_it(in_img)
    # myplot.plot(img)

    return img




def do_it(input, output):
    clip = VideoFileClip(input).subclip(t_start=37)
    clip = clip.set_duration(1)
    clip = clip.fl_image(process_images)
    clip.write_videofile(output, progress_bar=True, audio=False)








do_it(input='project_corrected.mp4', output='./temp_output/project_test.mp4')

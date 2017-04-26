import collections
import numpy as np
import detect_multiple
import myplot



buffer = collections.deque(maxlen=10)


def do_it(in_img, rectangles):
    global buffer
    buffer.append(rectangles)
    # print(buffer)

    out_img = np.copy(in_img)
    out_img = detect_multiple.do_it(buffer, out_img)
    # myplot.plot(out_img)
    return out_img




import collections
import numpy as np
import detect_multiple
import myplot



buffer = collections.deque(maxlen=10)

x_threshold = 500


def do_hog(rectangle_lists):
    global buffer

    for lst in rectangle_lists:
        # Skip any empty lists
        if lst == []: continue
        # if lst is None: continue


        filtered = []
        for rect in lst:
            # Unpack the points
            start, end = rect
            startx, starty = start
            # endx, endy = end

            if startx > x_threshold:
                filtered.append(rect)

        # Append the flitered list of 'hot' rectangles.
        buffer.append(filtered)

    return buffer



def do_search(rectangles):
    global buffer

    # filtered = []

    # for rect in rectangles:
    #     # Unpack the points
    #     start, end = rect
    #     startx, starty = start
    #     # endx, endy = end
    #
    #     if startx > x_threshold:
    #         filtered.append(rect)

    # Append the flitered list of 'hot' rectangles.
    buffer.append(rectangles)

    return buffer


# def make_return_list():
#     ret_list = []
#     for item in buffer:
#         ret_list.append(item)
#
#     return ret_list
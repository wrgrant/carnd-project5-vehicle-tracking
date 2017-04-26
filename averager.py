import collections
import numpy as np
import detect_multiple
import myplot



buffer = collections.deque(maxlen=10)



def do_search(rectangles):
    global buffer

    # Append the flitered list of 'hot' rectangles.
    buffer.append(rectangles)

    return buffer
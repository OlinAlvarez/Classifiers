import cv2
import numpy as np
from matplotlib import pyplot as plt
import data_utils

img = data_utils.get_random_die()
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)

while True:
    cv2.imshow('Color input image', img)
    cv2.imshow('Histogram equalized', img_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

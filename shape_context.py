import cv2
import matplotlib.pyplot as plt
import data_utils
#frame = cv2.imread('dice_test_data/die0155.jpg')
frame = data_utils.get_random_die()
print type(frame)

while True:
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
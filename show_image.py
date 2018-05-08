import random
import cv2
import numpy as np
import data_utils
import DiceClassifier as dc


classifier = dc.DiceClassifier()

frame = data_utils.get_random_die()
#frame = cv2.imread('dice_test_data/die0155.jpg')
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(3.0, (8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
gray = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 150, 200)
im, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


minLineLength = 50
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
print type(lines)

for i in range(0, len(lines) - 1):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 2)

print len(contours)
contours = [ c for c in contours if len(c) > 30]
print len(contours)
#edges = cv2.cvtColor(edges, cv2.COLOR)
print type(edges)

cv2.drawContours(final, contours, -1, (255, 0, 0), 2)

while True:

    cv2.imshow('die', edges)
    cv2.imshow('final', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


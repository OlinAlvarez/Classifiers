import cv2
import utils
import data_utils
import math
import p_DiceClassifier as DC
import numpy as np

#frame = cv2.imread('dice_test_data/dice1521499707100.jpg')
#frame = cv2.imread('dice_test_data/die0155.jpg')
#frame = cv2.imread('gate.jpg')
#frame = cv2.imread('front9.jpg')

frame = data_utils.get_random_die()
classifier = DC.p_DiceClassifier()

struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
morph = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,struct)
lab = cv2.cvtColor(morph, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(3.0, (10, 10))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))

final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
gedges = cv2.Canny(gray, 100, 200)
edges = cv2.Canny(blurred, 100, 200)


im, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
gim, gcontours, _ = cv2.findContours(gedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

gboxes = [cv2.boundingRect(c) for c in gcontours]
boxes = [cv2.boundingRect(c) for c in contours]

#rois = [b for b in boxes if b[2] * b[3] > 400 and math.fabs(b[2] - b[3]) < 20]
grois = [b for b in gboxes if b[2] * b[3] > 400 and '''math.fabs(b[2] - b[3]) < 20''']
#grois.append((346,195,142,128))
'''
for x, y, w, h in rois:
    cv2.rectangle(final, (x, y), (x+w, y+h), utils.colors["blue"], 1)
    '''
for x, y, w, h in grois:
    cv2.rectangle(final, (x, y), (x+w, y+h), utils.colors["red"], 1)


dice = classifier.classify(frame, grois)

for x, y, w, h in dice:
    cv2.rectangle(final, (x, y), (x+w, y+h), utils.colors["magenta"], 2)

while True:
    cv2.imshow('frame',frame)
    cv2.imshow('gedges', gedges)
    cv2.imshow('edges', edges)
    cv2.imshow('gray',gray)
    cv2.imshow('final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

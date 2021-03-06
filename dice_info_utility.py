import math
import data_utils
import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import DiceClassifier as DC
import time
import glob
from sklearn.externals import joblib

#frame = data_utils.get_random_die()

classifier = DC.DiceClassifier()

def save_pos(win):
    cv2.imwrite('pos_dice/' + str(time.time()) + 'dice.jpg',win)


def save_neg(win):
    cv2.imwrite('neg_images/' + str(time.time()) + 'dice.jpg',win)


frames = []
#picspath =  'sdsu_pics/dice/*.jpg'
picspath =  'bumble_share/images/*.jpg'
#picspath = 'dice_test_data/*.jpg'
for img in glob.glob((picspath)):
    frames.append(cv2.imread(img))

print len(frames)
not_found = []
ctr = 0
for frame in frames:
    clone = frame.copy()

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(3.0, (8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    gedges = cv2.Canny(gray,100,200)
    edges = cv2.Canny(blurred,100,200)

    gim, gcontours, _ = cv2.findContours(gedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gboxes = [cv2.boundingRect(c) for c in gcontours]
    grois = [b for b in gboxes if b[2] * b[3] > 400 and math.fabs(b[2] - b[3]) < 30]
    
    for x,y,w,h in grois:
        cv2.rectangle(frame, (x,y),(x+w,y+h),utils.colors["red"], 3)

    dice = classifier.classify(frame,grois)

    if len(dice) == 0:
        not_found.append(frame)

    for x, y, w, h in dice:
        cv2.rectangle(final, (x, y), (x+w, y+h), utils.colors["magenta"], 2)
    plt.imshow(final)
    #plt.imshow(gim)
    plt.show()

    print len(dice)
    print 'die :: ' + str(ctr)
    ctr += 1
    all = False

    for x, y, w, h in dice:
        portion = clone[y:y+h, x:x+w]
        plt.imshow(portion)

        if all:
            save_neg(portion)

        else:
            plt.show()
            choice = raw_input('is this a false positive? [y/n]')

            if choice == 'y':
                save_neg(portion)
            if choice == 'n':
                save_pos(portion)
            if choice == 'a':
                save_neg(portion)
                all = True
            if choice == 's':
                break
'''
ctr2= 1
for img in not_found:
    cv2.imwrite('not_found/die' + str(ctr2) + '.jpg',img)
    ctr2 += 1
'''

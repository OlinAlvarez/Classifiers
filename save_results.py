import math
import utils
import cv2
import matplotlib.pyplot as plt
import p_DiceClassifier as DC
import glob

classifier = DC.p_DiceClassifier()


frames = []

for img in glob.glob(('pos_dice/*.jpg')):
    frames.append(cv2.imread(img))

print len(frames)

ctr = 0
dice_found = 0
for frame in frames:
    clone = frame.copy()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

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
    '''
    for x,y,w,h in grois:
        cv2.rectangle(final, (x,y),(x+w,y+h),utils.colors["green"], 2)
    '''
    dice = classifier.classify(frame,grois)
    for x, y, w, h in dice:
        cv2.rectangle(frame, (x, y), (x+w, y+h), utils.colors["magenta"], 2)
    if len(dice) > 0:
        cv2.imwrite('test_results/die' + str(ctr) + '.jpg', frame)
        dice_found += 1
    ctr += 1
print dice_found

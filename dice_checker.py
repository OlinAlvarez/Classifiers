'''
This file makes sure that the crop dice are all recognized as positives.

'''
import cv2
import glob
import p_DiceClassifier as dc

classifier = dc.p_DiceClassifier()
images = []

for img in glob.glob('pos_dice/*.jpg'):
    images.append(cv2.imread(img))

positives = 0
total_dice = 0

for img in images:
    while True:
        cv2.imshow('img',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_dice += 1
    '''
    if classifier.is_die(img):
        cv2.imshow('die',img)
        positives += 1
    '''
print positives, total_dice, positives / total_dice

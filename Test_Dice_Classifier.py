import DiceClassifier as dc
import numpy as np
import cv2
import glob
import DicePreprocess as dp
import matplotlib.pyplot as plt


classifier = dc.DiceClassifier()
preprocessor =  dp.DicePreprocessor()

imgs = []
for img in glob.glob('dice_test_data/*.jpg'):
    imgs.append(cv2.imread(img))

print len(imgs)
for img in imgs:
    irs = preprocessor.get_interest_regions(img)  #Interest Regions
    dice = classifier.classify(img,irs)

    for ir in irs:
        cv2.rectangle(img,(ir[0],ir[1]),(ir[0] + ir[2] , ir[1] + ir[3]), (0,123,123), 2)
    for die in dice:
        cv2.rectangle(img,(ir[0],ir[1]),(ir[0] + ir[2] , ir[1] + ir[3]), (0,255,0), 2)

    plt.imshow(img)
    plt.show()






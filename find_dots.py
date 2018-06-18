import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

lower = np.array([0, 80, 80], dtype='uint8')
upper = np.array([170, 255, 255], dtype='uint8')
dice = []
for img in glob.glob('pos_dice/*.jpg'):
    dice.append(cv2.imread(img))
print len(dice)

for die in dice:
    mask = cv2.inRange(die,  lower,  upper)
    output = cv2.bitwise_and(die,  die,  mask=mask)
    imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    flag, binary_image = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary_image, 50, 150)
    #circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 5)
    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is None:
        circles = [(1, 1, 1)]

    if len(circles) > 0:
        circle_count = len(circles)
        print "number of circles,", circle_count
        #cv2.putText(die,str(circle_count),cv2.FONT_HERSHEY_SIMPLEX,(10,10),1,(0,255,122),2)
        for x, y, r in circles:
            print x,y,r
            cv2.circle(die, (x, y), r, (0, 255, 0), 4)
    plt.imshow(die)
    plt.show()





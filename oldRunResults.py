import cv2
import glob
import matplotlib.pyplot as plt
import clahe_filter
imgs = []
for img in glob.glob('fourthRun/*.jpg'):
    imgs.append(cv2.imread(img))

clone_imgs = [img.copy() for img in imgs]
#imgs =  [img[0:480,  110:570,:] for img in imgs]
imgs =  [clahe_filter.filter(img) for img in imgs]

for i in range(len(imgs)):
    while True:
        cv2.imshow('frame',imgs[i])
        cv2.imshow('clone',clone_imgs[i])


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            exit(0)

cv2.destroyAllWindows()

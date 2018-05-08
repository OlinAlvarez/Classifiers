import cv2
import glob

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 1.0, (640,480))

imgs = []
for img in glob.glob('found_dice/*.jpg'):
    imgs.append(cv2.imread(img))

for img in imgs:
    out.write(img)

import cv2
import glob


clahe = cv2.createCLAHE(3.0, (8,8))
imgs= []

for img in glob.glob('pos_dice/*.jpg'):
    imgs.append(cv2.imread(img))
i = 0
for img in imgs:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite('clahe_dice_pos/die_' + str(i) + '.jpg',final)
    i+=1

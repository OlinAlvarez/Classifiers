'''
This file takes in two image arguments and then finds motion vectors for the different blocks.
'''
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
x = 246
y = 246
width = 108
height = 108
p = 16
x = 246
y = 246
width = 108
height = 108

def search(ref_block, tar_img,x1,y1):
    c_w, c_h, _ = np.shape(tar_img)

    ystart =  max( 0, y1 - p)
    xstart =  max( 0, x1 - p)
    yend =   min(c_h, y1 + width + p)
    xend =   min(c_w, x1 + height + p)

    low = sys.maxint
    curr_sum = 0
    mv = []
    for i in range(ystart, yend - height):
        for j in range(xstart, xend - width):
            if i + width < c_w and j + height < c_h:
                temp_block = tar_img[i: i + width, j: j + height]
                curr_sum = mse(ref_block, temp_block)
                if curr_sum < low:
                    low = curr_sum
                    mv = [i,j]
    return mv

def mse(A, B):
	if A.shape != B.shape:
		return None
	return np.square(np.subtract(A, B)).mean()

print len(sys.argv)
print sys.argv
if len(sys.argv) != 3:
    print "error invalid arguments"
    sys.exit()

ref_img = cv2.imread(sys.argv[1])
tar_img = cv2.imread(sys.argv[2])

grey_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
grey_tar = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)

ref_block = grey_ref[y: y + height, x: x + width]

plt.imshow(ref_block)
plt.show()

vect = search(ref_block, tar_img,x,y)
win = grey_ref[vect[0]: vect[0] + width, vect[1]: vect[1] + height]

#plt.imshow(win)
#plt.show()

cv2.line(ref_img, (vect[0],vect[1]), (x,y), (0,125,135), 2)
cv2.rectangle(ref_img, (x,y), (x + width, y + height), (0,140,0), 2)
cv2.rectangle(tar_img, (vect[0],vect[1]),(vect[0] + width,vect[1] + height), (0,140,0), 2)
_, ax =  plt.subplots(1,2)
ax[0].imshow(ref_img)
ax[1].imshow(tar_img)
plt.show()


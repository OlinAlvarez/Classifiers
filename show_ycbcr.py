import cv2

img = cv2.imread('Ducky.jpg')
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

y, cr, cb = cv2.split(ycrcb)

while True:
    cv2.imshow('frame',img)
    cv2.imshow('y',y)
    cv2.imshow('cb',cb)
    cv2.imshow('cr',cr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

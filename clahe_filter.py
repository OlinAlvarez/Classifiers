import cv2
import matplotlib.pyplot as plt
def filter(frame):
    clone = frame.copy()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(3.0, (8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

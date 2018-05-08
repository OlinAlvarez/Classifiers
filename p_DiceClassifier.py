from sklearn.externals import joblib
import cv2
import matplotlib.pyplot as plt

class p_DiceClassifier():

    def __init__(self):
        self.svm = joblib.load('Dice_SVM_std.pkl')
        self.hog = self.get_hog()
        self.dims = (144, 144)

        print type(self.hog)
        print type(self.svm)

    def get_hog(self):
        minDim = 80
        blockSize = (8,8)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = -1
        histogramNormType = 0
        L2HysThreshold = 2.1e-1
        gammaCorrection = 0
        nlevels = 64
        dims = (144,144)
        return cv2.HOGDescriptor(dims, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    #rois = regions of interest

    def classify(self, frame, rois):

        true_dice = []
        for box in rois:
            x,  y,  w,  h = box
            window = frame[y:y + h,  x:x + w,:]
            window = cv2.resize(window,  self.dims)
            feat = self.hog.compute(window)
            prob = self.svm.predict_proba(feat.reshape(1,  -1))[0]
            if prob[1] > .8:
                true_dice.append(box)
        return true_dice

    def sliding_window(self,frame):
        window_width = 600/4
        window_height = 480/4
        step_size = window_width
        true_dice = []
        for y in range(0,frame.shape[0]-step_size,window_height):
            for x in range(0,frame.shape[1]-step_size,window_width):
                window = frame[y:y + window_height, x:x + window_width, :]
                window = cv2.resize(window, self.dims)
                feat = self.hog.compute(window)
                prob = self.svm.predict_proba(feat.reshape(1, -1))[0]
                if prob[1] > .1:
                    true_dice.append((x,y,window_width,window_height))
        return true_dice

    def is_die(self,frame):
        try:
            feat = self.hog.compute(frame)
            prob = self.svm.predict_proba(feat.reshape(1,  -1))[0]
            if prob[1] > .3:
                return True

        except cv2.error as e:
            return False
            print e

        return False

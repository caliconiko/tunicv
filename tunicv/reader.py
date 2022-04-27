import cv2
import numpy as np


class Reader:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(str(self.path))

    def read(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.stroke = cv2.threshold(self.gray, 254, 255, cv2.THRESH_OTSU)
        self.distance_transformed = cv2.distanceTransform(self.stroke, cv2.DIST_L2, 0)

        # double stroke dimensions
        self.double_stroke = cv2.resize(self.stroke, (0, 0), fx=2, fy=2)

        self.skeletonized = Reader.skeletonize(self.double_stroke)
        # threshold again
        _, self.skeletonized = cv2.threshold(self.skeletonized, 254, 255, cv2.THRESH_OTSU)

        cv2.imshow("Image", self.skeletonized)
        cv2.waitKey(0)

    # hat tip to https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d
    @staticmethod
    def skeletonize(img):
        """ OpenCV function to return a skeletonized version of img, a Mat object"""

        #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

        img = img.copy() # don't clobber original
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        return skel

    @staticmethod
    def distance_transform(img, dist_type=cv2.DIST_L2, mask_size=0):
        return cv2.distanceTransform(img, dist_type, mask_size)

    @staticmethod
    def morph(img, kernel_size=2, morph=cv2.MORPH_BLACKHAT):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return cv2.morphologyEx(img, morph, kernel)

    @staticmethod
    def morph_func(img, func, kernel_size=2, iterations=1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return func(img, kernel, iterations=iterations)

    @staticmethod
    def find_contours(img, retr=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_SIMPLE):
        return cv2.findContours(img, retr, approx)

    @staticmethod
    def mask_contours(cnts, img, color = 255):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, cnts, -1, color, -1, cv2.LINE_AA)
        return mask
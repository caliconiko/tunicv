from pickletools import uint1
import cv2
import numpy as np


class Reader:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(str(self.path))

    def read(self):
        # gray scale it
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # get only the writing
        _, stroke = cv2.threshold(gray, 254, 255, cv2.THRESH_OTSU)
        # get stroke width of diffrent parts of image
        distance_transformed = cv2.distanceTransform(stroke, cv2.DIST_L2, 0)
        # get the max stroke width
        max_stroke = np.max(distance_transformed)
        # make the writing as thin as possible without destroying it
        eroded_stroke = Reader.morph_func(stroke, cv2.erode, int(max_stroke/2))

        RESCALE_FACTOR = 2
        # make image biggar
        double_stroke = cv2.resize(eroded_stroke, (0, 0), fx=RESCALE_FACTOR, fy=RESCALE_FACTOR)

        # skelotonize to make it even thinner
        skeletonized = Reader.skeletonize(double_stroke)
        # threshold again just in case
        _, skeletonized = cv2.threshold(skeletonized, 254, 255, cv2.THRESH_OTSU)
        # get rid of holes
        closed_skeleton = Reader.morph(skeletonized, kernel_size=5, morph=cv2.MORPH_CLOSE)
        
        # squish the image horizontally to find the lines
        hor_avg = cv2.normalize(np.average(closed_skeleton, axis=1), None, 0, 1, cv2.NORM_MINMAX)
        # binarize it just in case
        _, hor_avg_thresh = cv2.threshold(hor_avg, 0.5, 1, cv2.THRESH_BINARY)
        # pad margins to make the contours detectable
        hor_avg_pad = np.pad(np.array(hor_avg_thresh, dtype=np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=0)
        # find contours
        contours, _ = Reader.find_contours(hor_avg_pad)

        line_ys = []
        for contour in contours:
            # get center of contour
            center = np.average(contour, axis=0)
            # store y part of center
            line_ys.append(center[1])
            
        # thick version of line finding to make it human visible
        thick_hor_avg = cv2.resize(hor_avg_thresh, (0, 0), fx=200, fy=1)

        print(np.where(hor_avg > 0.5))
        cv2.imshow("image", thick_hor_avg)
        cv2.imshow("image1", closed_skeleton)
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
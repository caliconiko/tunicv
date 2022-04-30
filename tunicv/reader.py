import enum
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
        double_stroke = cv2.resize(stroke, (0, 0), fx=RESCALE_FACTOR, fy=RESCALE_FACTOR)

        # skelotonize to make it even thinner
        skeletonized = Reader.skeletonize(double_stroke)
        # threshold again just in case
        _, skeletonized = cv2.threshold(skeletonized, 254, 255, cv2.THRESH_OTSU)
        # get rid of holes
        good_skeleton = Reader.morph(skeletonized, kernel_size=2, morph=cv2.MORPH_CLOSE, iterations=2)
        
        # squish the image vertically to find the lines
        ver_avg = cv2.normalize(np.average(good_skeleton, axis=1), None, 0, 1, cv2.NORM_MINMAX)
        # binarize it just in case
        _, ver_avg_thresh = cv2.threshold(ver_avg, 0.3, 1, cv2.THRESH_BINARY)
        # pad margins to make the contours detectable
        ver_avg_pad = np.pad(np.array(ver_avg_thresh, dtype=np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=0)
        # find globs in vertical profile
        contours, _ = Reader.find_contours(ver_avg_pad)

        # store the coordinates of the lines
        line_ys = []
        for contour in contours:
            # get center of contour
            center = np.average(contour, axis=0)
            # store y part of center
            line_ys.append(int(center[0][1])-1)
        # sort the coordinates from smol to beeg
        line_ys.sort()
        
        # get coordinates of the lines between the lines
        between_ys = [(y+line_ys[i+1])//2 for i, y in enumerate(line_ys[:-1])]
        between_ys.insert(0, 0)
        between_ys.append(good_skeleton.shape[0]-1)

        # store em into a list of objects
        lines:list[Line] = []
        for i, line_y in enumerate(line_ys):
            line = Line(good_skeleton, line_y, between_ys[i], between_ys[i+1])
            lines.append(line)
  
        # thick version of line finding to make it human visible
        thick_ver_avg = cv2.resize(ver_avg_thresh, (0, 0), fx=200, fy=1)

        debug_stroke = cv2.cvtColor(double_stroke, cv2.COLOR_GRAY2BGR)
        debug_skeleton = cv2.cvtColor(good_skeleton, cv2.COLOR_GRAY2BGR)
        for l in lines:
            cv2.line(debug_stroke, (0, l.bottomline), (debug_stroke.shape[1], l.bottomline), (255, 0, 255), 1)
            cv2.line(debug_skeleton, (0, l.bottomline), (debug_stroke.shape[1], l.bottomline), (255, 0, 255), 1)
            
            for w in l.words:
                cv2.rectangle(debug_stroke, (w.left, l.topline), (w.right, l.bottomline), (0, 255, 0), 1)
                cv2.rectangle(debug_skeleton, (w.left, l.topline), (w.right, l.bottomline), (0, 255, 0), 1)

        cv2.imshow("image", thick_ver_avg)
        cv2.imshow("image1", debug_stroke)
        cv2.imshow("image2", debug_skeleton)

        # show lines for debugging
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
    def morph(img, kernel_size=2, morph=cv2.MORPH_BLACKHAT, iterations=1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return cv2.morphologyEx(img, morph, kernel, iterations=iterations)

    @staticmethod
    def morph_func(img, func, shape=cv2.MORPH_RECT, kernel_size=2, iterations=1):
        kernel = cv2.getStructuringElement(shape, (kernel_size,kernel_size))
        return func(img, kernel, iterations=iterations)

    @staticmethod
    def find_contours(img, retr=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_SIMPLE):
        return cv2.findContours(img, retr, approx)

    @staticmethod
    def mask_contours(cnts, img, color = 255):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, cnts, -1, color, -1, cv2.LINE_AA)
        return mask

class Line:
    """class to store data of a line"""
    def __init__(self, skel_image:np.array, midline:int, topline:int, bottomline:int) -> None:
        self.skel_image = skel_image
        self.midline = midline
        self.topline = topline  
        self.bottomline = bottomline

        self.skel_line = self.skel_image[self.topline:self.bottomline, :]
        self.words = self.get_words()

    def get_words(self):
        # get horizontal profile of the line
        squish_hor = np.average(self.skel_line, axis=0)
        _, squish_hor_thresh = cv2.threshold(squish_hor, 2, 255, cv2.THRESH_BINARY)

        squish_hor_thresh_pad = np.pad(np.array(squish_hor_thresh, dtype=np.uint8), (1, 1), 'constant', constant_values=0)
        # find globs in horizontal profile
        contours, _ = Reader.find_contours(squish_hor_thresh_pad)
        
        words = []
        for contour in contours:
            contour = contour-np.array([[1, 1]])
            # get span of contour
            _, x1, _, w = cv2.boundingRect(contour)
            x2 = x1+w

            # get part of image included in span
            midline_intersection = self.skel_image[self.midline, x1:x2]

            if np.any(midline_intersection>0):
                word = Word(self, x1, x2)
                words.append(word)

        return words[::-1]

class Word:
    """class to store data of a word"""
    def __init__(self, line:Line, left:int, right:int) -> None:
        self.line = line
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.left}-{self.right}"
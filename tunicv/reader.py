import cv2

class Reader:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(str(self.path))

    def read(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.stroke = cv2.threshold(self.gray, 254, 255, cv2.THRESH_OTSU)
        self.distance_transformed = cv2.distanceTransform(self.stroke, cv2.DIST_L2, 0)
        cv2.imshow("Image", self.distance_transformed)
        cv2.waitKey(0)
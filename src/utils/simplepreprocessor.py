import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter # 데이터를 리사이징할 때 사용하는 알고리즘
        # 여기서는 cv2의 인터폴레이션 알고리즘을 디펄트로 사용함

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height),
                interpolation=self.inter)


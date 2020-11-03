import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):  # 입력 이미지별로 루핑

            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)  # 각 이미지를 로딩
            label = imagePath.split(os.path.sep)[-2]  # 경로에서 클래스를 저장

            if self.preprocessors is not None:

                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            # verbose 단위마다 업데이트 상황을 표시
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                                                      len(imagePaths)))
        return (np.array(data), np.array(labels))

    # t1_train_anomaly_detector.py 때문에 추가된 부분
    def quantify_image(self, image, bins):
        # compute a 3D color histogram over the image and normalize it
        hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist

    def load_quantified_dataset(self, imagePaths, bins=(4, 6, 3)):
        data = []

        for imagePath in imagePaths:
            # load the image and convert it to the HSV color space
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # quantify the image and update the data list
            features = self.quantify_image(image, bins)
            data.append(features)

            # return our data list as a NumPy array
        return np.array(data)

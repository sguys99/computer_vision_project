from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat = None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)
        # format이 None이면 케라스 내부의 'channel last'가 적용됨
        # 관련설정은 /.keras/keras.json을 확인하면 됨
        # 필요한 이유 mxnx3 꼴의 데이터로 읽어야 하기 때문
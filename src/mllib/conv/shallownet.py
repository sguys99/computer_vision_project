from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
# from tensorflow.keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # width : 이미지의 가로, 컬럼 크기
        # height : 이미지의 세로, row 크기
        # 데이터셋 전체 클래스의 크기
        # 예를들어 cifar-10은 classes=10
        inputShape = (height, width, depth)

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(units=classes ))
        model.add(Activation('softmax'))

        return model
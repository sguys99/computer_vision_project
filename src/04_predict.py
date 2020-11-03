
from utils import ImageToArrayPreprocessor, SimplePreprocessor, SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required= True, help = 'path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to pre-trained model')
args = vars(ap.parse_args())

print('[INFO] sampling images...')
imagePaths = np.array(list(paths.list_images(args['dataset']))) #np.array가 없으면 동작안함. 확인요
idxs = np.random.randint(0, len(imagePaths), size = (10, ))# 10개 추출
imagePaths = imagePaths[idxs] #10개 이미지 경로 랜덤 추출

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths=imagePaths)
data = data.astype('float')/255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args['model'])

print('[INFO] predicting')
classLabels = ['cat', 'dog', 'panda']

preds = model.predict(data, batch_size = 32).argmax(axis = 1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread((imagePath))
    cv2.putText(image, text="Label: {}".format(classLabels[preds[i]]),
                org = (10, 30), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color=(0, 255, 0), thickness=1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)



from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import ImageToArrayPreprocessor, SimplePreprocessor, SimpleDatasetLoader
from mllib.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required= True, help = 'path to input dataset')
ap.add_argument('-m', '--model', required= True, help = 'path to output model')
ap.add_argument('o', '--output', help = 'path to train results')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths=imagePaths, verbose = 500)

data = data.astype('float')/255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25,
                                                      random_state=42)

y_train = LabelBinarizer().fit_transform(y_train) #문자열을 원핫으로.. 단 정수로..
y_test = LabelBinarizer().fit_transform(y_test)
# labels은 폴더명인 cat, dog, panda 중 하나이다. 그리고 알파벳 순이므로 cat, dog, panda 순일 것이다.

print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth= 3, classes=3)
model.compile(loss = 'categorical_crossentropy', optimizer = opt,
              metrics = ['accuracy'])

print('[INFO] training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=32, epochs = 100, verbose = 1)

print('[INFO] serializing netowork...')
model.save(args['model'])
# 중요 : --model 뒤에 경로를 입력하면 .pb 타입으로 저장됨
#        --model 파일명.hdf5로 입력하명 hdf5 파일로 저장됨

print('[INFO] evaluating network...')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),
        target_names=["cat", "dog", "panda"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# plt.savefig(args["output"])
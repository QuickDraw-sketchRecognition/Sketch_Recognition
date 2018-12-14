
print('Initializing the execution')

print('Importing the libraries')
import os
import json
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet,MobileNetV2
# keras.applications.mobilenet_v2
from tensorflow.keras.applications.mobilenet import preprocess_input
start = dt.datetime.now()

# Defining the required paths for the datasets
DP_DIR = '/home/akhilesh_narapareddy/apm_train_data/shuffled_csvs/'
INPUT_DIR = '/home/akhilesh_narapareddy/apm_train_data/'

BASE_SIZE = 256

# Number of files to be considered from the shuffled csvs
NCSVS = 100

# Total classes to be predicted
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

print('Defining the assessment metrics')
def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)

def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# Setting the parameters for training
STEPS = 2300
EPOCHS = 70
size = 128
batchsize = 600

print('Parameters used for training')
print('Steps:',STEPS)
print('Epochs:',EPOCHS)
print('Size:',size)
print('Batchsize:',batchsize)

print('Initializing the model')
model = MobileNet(input_shape = (size,size,1), alpha = 1.,weights = None,classes = NCATS)
model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy',
              metrics = [categorical_crossentropy,categorical_accuracy,top_3_accuracy])

print(model.summary())

## Defining the functions for converting the drawing coordinates to images  
def draw_cv2(raw_strokes,size = 256, lw=6,time_color = True):
    img = np.zeros((BASE_SIZE,BASE_SIZE),np.uint8)
    for t,stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0])-1):
            color = 255 - min(t,10)*13 if time_color else 255
            _ = cv2.line(img,(stroke[0][i], stroke[1][i]),
                              (stroke[0][i+1], stroke[1][i+1]),color,lw)
    if size != BASE_SIZE:
        return cv2.resize(img,(size,size))
    else:
        return img
    
def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

print('Creating validation dataset')
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

print('Creating train_datagen object')
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))


x, y = next(train_datagen)

# model.load_weights('model_final.h5')
print('Model training')
callbacks = [
    ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.75, patience=3, min_delta=0.001,
                          mode='max', min_lr=1e-5, verbose=1),
    ModelCheckpoint('model_vf.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,
                    save_weights_only=True),
]

hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)

print('Calculating MAP')
valid_predictions = model.predict(x_valid, batch_size=128, verbose=2)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))

print('Creating test files')
INPUT_DIR_TEST = '/home/akhilesh_narapareddy/apm_test_data/'
test = pd.read_csv(os.path.join(INPUT_DIR_TEST, 'test_simplified.csv'))
test_split = np.array_split(test, 10)
final_test = {}
for i in range(10):
    final_test[i] = df_to_image_array_xd(test_split[i], size)

print('Predicting from test files')
test_final = {}
for i in range(10):
    test_final[i] = model.predict(final_test[i], batch_size=128, verbose=2)

test_vf = np.concatenate((test_final[0],test_final[1],test_final[2],test_final[3],test_final[4],test_final[5],test_final[6],test_final[7],test_final[8],test_final[9]),axis = 0)
top3 = preds2catids(test_vf)
top3.head()
top3.shape

# Please change the directory accordingly
cats = pd.read_csv('/home/akhilesh_narapareddy/apm_test_data/categories_list.csv',header = None)
cats = cats[0].tolist()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
top3cats.head()
top3cats.shape

print('Creating the submissions')
test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('kaggle_submission_vf1.csv', index=False)
submission.head(20)
submission.shape

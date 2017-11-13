from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_data():
    # load data
    print "loading data ..."
    df = pd.read_csv('/Users/f617602/sundeep/projects/dog_breed_data/labels.csv')
    df.head()

    n = len(df)
    breed = set(df['breed'])
    n_class = len(breed)
    class_to_num = dict(zip(breed, range(n_class)))
    num_to_class = dict(zip(range(n_class), breed))

    width = 299
    X = np.zeros((n, width, width, 3), dtype=np.uint8)
    y = np.zeros((n, n_class), dtype=np.uint8)

    print "loading training images ..."
    train_image_dir_root = "/Users/f617602/sundeep/projects/dog_breed_data/train"
    for i in tqdm(range(n)):
        image_name = "{}.jpg".format(df["id"][i])
        image_path = os.path.join(train_image_dir_root, image_name)
        if os.path.exists(image_path):
            X[i] = cv2.resize(cv2.imread(image_path), (width, width))
            y[i][class_to_num[df['breed'][i]]] = 1

    print X.shape
    print y.shape
    return X, y, n_class, width, breed, class_to_num


def load_test_data(width):
    print "loading test data ..."
    df2 = pd.read_csv('sample_submission.csv')
    n_test = len(df2)
    X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
    for i in tqdm(range(n_test)):
        X_test[i] = cv2.resize(cv2.imread('/Users/f617602/sundeep/projects/dog_breed_data/test/%s.jpg' % df2['id'][i]), (width, width))

    return X_test, df2


def extract_features_from_model(MODEL, image_data, width):
    print "extract_features_from_model ..."
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(image_data, batch_size=64, verbose=1)
    return features


def aggregate_features(image_data, width):
    print "aggregate_features ..."
    inception_features = extract_features_from_model(InceptionV3, image_data, width)
    xception_features = extract_features_from_model(Xception, image_data, width)
    features = np.concatenate([inception_features, xception_features], axis=-1)
    return features


def fit_model(features, y, n_class):
    print "fit_model ..."
    inputs = Input(features.shape[1:])
    x = inputs
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(features, y, batch_size=128, epochs=1, validation_split=0.1)
    return model, history


def visualize_metrics(h):
    print "visualize_metrics"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')


def main():
    # handle training data
    X, y, n_class, width, breed, class_to_num = load_data()
    features = aggregate_features(X, width)
    model, h = fit_model(features, y, n_class)
    visualize_metrics(h)

    # handle testing data
    X_test, df2 = load_test_data(width)
    features_test = aggregate_features(X_test, width)
    y_pred = model.predict(features_test, batch_size=128)
    for b in breed:
        df2[b] = y_pred[:, class_to_num[b]]
    df2.to_csv('pred.csv', index=None)


main()

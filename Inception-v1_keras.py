from inception_module import *
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import cv2

class Inception:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # block1(head)
        conv1 = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(input)
        conv1 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(conv1)

        # block2
        conv2 = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu')(conv1)
        conv2 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(conv2)

        # block3
        conv3 = inception_module(conv2, 64, 96, 128, 16, 32, 32, name='inception_3a')
        conv3 = inception_module(conv3, 128, 128, 192, 32, 96, 64, name='inception_3b')
        conv3 = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_3_3x3/2')(conv3)

        # block4

        conv4 = inception_module(conv3, 192, 96, 208, 16, 48, 64, name='inception_4a')
        conv4 = inception_module(conv4, 160, 112, 224, 24, 64, 64, name='inception_4b')
        conv4 = inception_module(conv4, 128, 128, 256, 24, 64, 64, name='inception_4c')
        conv4 = inception_module(conv4, 112, 144, 288, 32, 64, 64, name='inception_4d')
        conv4 = inception_module(conv4, 256, 160, 320, 32, 128, 128, name='inception_4e')
        conv4 = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_4_3x3/2')(conv4)

        # block5
        conv5 = inception_module(conv4, 256, 160, 320, 32, 128, 128, name='inception_5a')
        conv5 = inception_module(conv5, 384, 192, 384, 48, 128, 128, name='inception_5b')

        # Fully connected final layer
        averagepooling = GlobalAveragePooling2D(name='global_avg_pool_5_3x3/1')(conv5)
        dropout = Dropout(0.4)(averagepooling)
        output = Dense(10, activation='softmax', name='output')(dropout)

        #model
        model = keras.Model(inputs=input, outputs=output, name='Inception')
        model.compile(loss=['categorical_crossentropy'], optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

        model.summary()

        return model


if __name__ == "__main__" :

    EPOCH_ITER = 100
    BATCH_SIZE = 100

    # load cifar-10 training and validation sets
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.cifar10.load_data()
    x_train = x_train[0:2000, :, :, :]
    y_train = y_train[0:2000]

    x_valid = x_valid[0:500, :, :, :]
    y_valid = y_valid[0:500]

    # resize training images
    x_train = np.array([cv2.resize(img, (224, 224)) for img in x_train[:, :, :, :]])
    x_valid = np.array([cv2.resize(img, (224, 224)) for img in x_valid[:, :, :, :]])

    # transform targets to keras compatible format - 원핫인코딩으로
    y_train = np_utils.to_categorical(y_train)
    y_valid = np_utils.to_categorical(y_valid)

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')

    # preprocess data (영상이미지라서 255.0으로 나눠서 normalize한다)
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    Inception = Inception(input_shape=(224, 224, 3), num_classes=10)
    model = Inception.build_net()

    model.summary()
    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs",update_freq="batch")]

    history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=EPOCH_ITER,
                        callbacks=[callbacks],
                        validation_data=(x_valid, y_valid))

    

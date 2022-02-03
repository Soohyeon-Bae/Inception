import numpy as np
import tensorflow as tf
from utils import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision, CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import scipy.io


class inception_v1:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):
        inputs = keras.Input(shape=self.input_shape)

        # STEM
        stem_conv7x7 = Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
        # output size : 112x112x64
        stem_max3x3_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(stem_conv7x7)
        # output size : 56x56x64
        #stem_LRN1 = LRN(stem_max3x3_1, layer_name="stem_LRN1", depth_radius=7, bias=1, alpha=0.5, beta=0.5)
        # '3x3 reduce' stands for the number of 1×1 filters in the reduction layer used before the 3×3 convolutions
        # '3x3 reduce' = 64
        stem_conv1x1 = Conv2D(filters=64, kernel_size=1, strides=1, padding="valid", activation="relu")(stem_max3x3_1)
        stem_conv3x3 = Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(stem_conv1x1)
        # output size : 56x56x192
        #stem_LRN2 = LRN(stem_conv3x3, layer_name="stem_LRN2", depth_radius=7, bias=1, alpha=0.8, beta=0.5)
        stem_max3x3_2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(stem_conv3x3)
        # output size : 28x28x192

        # BODY
        module1 = naive_module(input=stem_max3x3_2, A_size=64, B1_size=96, B2_size=128, C1_size=16, C2_size=32, D2_size=32)
        # output size : 28x28x256
        module2 = naive_module(input=module1, A_size=128, B1_size=128, B2_size = 192, C1_size=32, C2_size=96, D2_size=64)
        # output size : 28x28x480
        module2_maxpool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(module2)
        # output size : 14x14x480
        module3 = naive_module(input=module2_maxpool, A_size=192, B1_size=96, B2_size=208, C1_size=16, C2_size=48, D2_size=64)
        # output size : 14x14x512
        module4 = naive_module(input=module3, A_size=160, B1_size=112, B2_size=224, C1_size=24, C2_size=64, D2_size=64)
        # output size : 14x14x512
        auxiliary_1 = auxiliary_classifier(module4, keep_prob=0.8)

        module5 = naive_module(input=module4, A_size=128, B1_size=128, B2_size=256, C1_size=24, C2_size=64, D2_size=64)
        # output size : 14x14x512
        module6 = naive_module(input=module5, A_size=112, B1_size=144, B2_size=288, C1_size=32, C2_size=64, D2_size=64)
        # output size : 14x14x528
        module7 = naive_module(input=module6, A_size=256, B1_size=160, B2_size=320, C1_size=32, C2_size=128, D2_size=128)
        # output size : 14x14x832
        auxiliary_2 = auxiliary_classifier(module7, keep_prob=0.8)

        module7_maxpool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(module7)
        # output size : 7x7x832
        module8 = naive_module(input=module7_maxpool, A_size=256, B1_size=160, B2_size=320, C1_size=32, C2_size=128, D2_size=128)
        # output size : 7x7x832
        module9 = naive_module(input=module8, A_size=384, B1_size=192, B2_size=384, C1_size=48, C2_size=128, D2_size=128)
        # output size : 7x7x1024

        # TAIL
        tail_avg7x7 = AvgPool2D(pool_size=(7,7), strides=(1,1), padding="valid")(module9) # 전체 영역에 대한 Pooling. Output은 결국 Feature의 개수만큼이다.
        # output size : 1x1x1024
        tail = Flatten()(tail_avg7x7)
        # output size : 1x1x1000
        tail = Dropout(rate=0.8)(tail)
        ### 데이터셋 바뀌면 변경할 부분 ###
        outputs = Dense(units=2, activation="softmax")(tail)

        model = keras.Model(inputs=inputs, outputs=(outputs, auxiliary_1, auxiliary_2))

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            ### 데이터셋 바뀌면 변경할 부분 (원래는 CategoricalCrossentropy) ###
            loss=[BinaryCrossentropy(), BinaryCrossentropy(), BinaryCrossentropy()],
            loss_weights=[1.0, 0.3, 0.3],
            metrics=["accuracy"]
        )

        return model

if __name__ == "__main__" :

    EPOCH_ITER = 100
    BATCH_SIZE = 64

    datagen = ImageDataGenerator()
    ### 데이터셋 바뀌면 변경할 부분 ###
    generator = datagen.flow_from_directory(
        directory="C:\\Users\\bsh\\Desktop\\Papers\\Inception\\cats_and_dogs_filtered",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    IMAGE_NUM = generator.samples
    steps_per_epoch = int(IMAGE_NUM/BATCH_SIZE)

    ### 데이터셋 바뀌면 변경할 부분 ###
    Inception = inception_v1(input_shape=generator.image_shape, num_classes=2)
    model = Inception.build_net()
    # load weights
    #model.load_weights(filepath="./inception_v1_weights/INCTION_12.hdf5")
    # callback lists
    callbacks = [
        # Save weights regularly
        keras.callbacks.ModelCheckpoint(filepath="./inception_v1_weights/INCTION_{epoch:02d}.hdf5"),
        # Recording Tensorboard
        keras.callbacks.TensorBoard(log_dir='./logs', write_images=True),
        # learning rate scheduler
        # keras.callbacks.LearningRateScheduler(scheduler)
    ]
    history = model.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCH_ITER,
        callbacks=[callbacks]
    )

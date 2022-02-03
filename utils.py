import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AvgPool2D, Flatten, Dropout, Dense



def auxiliary_classifier(input, keep_prob, verbose=True):
    """
    :param layer_name:
    :param is_training:
    :param input:
    :param keep_prob:
    :param label:
    :return:
    """
    avg_L = AvgPool2D(pool_size=(5,5), strides=(3,3), padding="valid")(input)
    con_L = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(avg_L)
    if verbose : con_L = BatchNormalization(axis=-1, momentum=0.9)(con_L)
    con_L = Activation(activation="relu")(con_L)
    con_L = Flatten()(con_L)
    FC_L = Dropout(rate=keep_prob)(con_L)
    FC_L = Dense(units=1024, activation="relu")(FC_L)
    FC_L = Dropout(rate=keep_prob)(FC_L)
    FC_L = Dense(units=102, activation="softmax")(FC_L)

    return FC_L

def naive_module(input, A_size, B1_size, B2_size, C1_size, C2_size, D2_size, pad="same"):
    """
    :param input : input feature map
    :param A_size: output channels of leftmost part in the naive module - conv1x1
    :param B1_size: output channels of second left side part in the naive module - conv1x1
    :param B2_size: output channels of second left side part in the naive module - conv3x3
    :param C1_size: output channels of third left side part in the naive module - conv1x1
    :param C2_size: output channels of third left side part in the naive module - conv5x5
    :param D2_size: output channels of rightmost side part in the naive module - conv1x1
    :return: concat
    """
    A1 = Conv2D(filters=A_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B1 = Conv2D(filters=B1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B2 = Conv2D(filters=B2_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(B1)
    C1 = Conv2D(filters=C1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    C2 = Conv2D(filters=C2_size, kernel_size=5, strides=(1,1), padding=pad, activation="relu")(C1)
    D1 = MaxPool2D(pool_size=(3,3), strides=(1,1), padding=pad)(input)
    D2 = Conv2D(filters=D2_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(D1)
    concat = Concatenate()([A1, B2, C2, D2])
    return concat


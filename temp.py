"""
Hinton氏の論文「Dynamic Routing Between Capsules」のカプセルネットワークをKerasで実装した.
おそらくバックエンドでTensorFlowでのみ動く.
試してはいないが, 別のバックエンドに適応させることは簡単である.

使い方:
    python capsulenet.py
    python capsulenet.py --epochs 50
    python capsulenet.py --epochs 50 --routings 3

結果:
    validation accuracy = 99.50% (epochs=20)
    validation accuracy = 99.66% (epochs=50)
    ※single GTX1070を使用
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from Keras.utils import combine_images
from PIL import Image
from capsulelayers import CapsulelLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')

def CapsNet(input_shape, n_class, routings):
    """
    MNISTに関するカプセルネットワーク

    :param input_shape: 入力データのshape(3次元で[width, height, channels]という形)
    :param n_class: クラスの数
    :param routings: routingを行う回数

    :return 2つのmodel (1つ目:学習用モデル, 2つ目:評価用モデル)
            `eval_model`というモデルも学習用としてしようすることもできる.
    """
    x = layers.Input(shape=input_shape)

    # 1層目: ただの2次元畳み込み層
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # 2層目: 活性化関数にsquash関数を用いた2次元畳み込み層で,[None ,num_capsule, dim_capsule]という形に変換する
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # 3層目: カプセル層 (routingアルゴリズムはここで行っている)
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # 4層目: ここはカプセルを"長さ"に変形するための補助レイヤーで, 教師データの形に合わせている.
    # tensorflowを使用している場合, ここは必要ありません.
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoderネットワーク
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])

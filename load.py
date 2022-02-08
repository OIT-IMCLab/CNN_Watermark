#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#CIFAR-10のデータセットのインポート
from tensorflow.keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
 
#CIFAR-10の正規化
from tensorflow.keras.utils import to_categorical
 
 
# 特徴量の正規化
X_train = X_train/255.
X_test = X_test/255.
 
# クラスラベルの1-hotベクトル化
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
 
# CNNの構築
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,InputLayer
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import random
import os
import tempfile
from tensorflow.keras import regularizers

def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(0)

from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend
import math

def custom(weights):
    w = tf.reduce_mean(weights,axis=3)
    w = tf.reshape(w,[288,1])

    x = np.random.rand(256,288)
    #x_arr /= x_arr.sum(axis=1)[:,np.newaxis]
    #x = np.random.normal(0,1,(256,576))
    x = tf.constant(x,dtype=tf.float32)
    #print(x.numpy)
    #x = tf.fill((288,1), 1.0) #秘密鍵行列xの簡単な例として、0.5の値を持つ行列を作成
    x_weights = tf.matmul(x,w)  #行列の積の結果、100行かける1列の行列になる

    x_weights = tf.sigmoid(x_weights) #100個のデータをまとめて1個にしてから、シグモイド関数
    
    arr = np.ones((256,1),dtype='float64')
    c = np.arange(0,256).reshape(256, 1)
    condition = c % 2 == 0
    c[condition] = 1
    c[~condition] = 0
    #print(c)

    #arr = tf.constant(c, dtype='float32')
    
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_weights,tf.keras.backend.cast_to_floatx(arr)))

    return 0.02 * loss

model = tf.keras.models.load_model('./original_200_256_all0.h5',custom_objects={'custom': custom})
model.summary()
#model.evaluate(X_test,Y_test)

layer = model.layers[1]
w = layer.get_weights()[0]
w = np.array(w)
#np.set_printoptions(np.inf)
print(w)
print(w.shape)
w = np.mean(w,axis=3)
w = np.reshape(w,[288,1])
#print(w)
x = np.random.rand(256,288)
#x /= x.sum(axis=1)[:,np.newaxis]
#x = np.random.normal(0,1,(256,576))
x_weights = np.matmul(x,w)
#print(x_weights)

# シグモイド関数の定義

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

x_weights = sigmoid(x_weights)
x_weights = x_weights.flatten()
print(x_weights)

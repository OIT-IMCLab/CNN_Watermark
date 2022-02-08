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

Y_test_tmp = Y_test

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
import tempfile
import os
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
    w = tf.reshape(w,[576,1])

    x = tf.constant(np.random.rand(10,576),dtype=tf.float32)

    x_weights = tf.matmul(x,w)  

    x_weights = tf.sigmoid(x_weights) 
    
    arr = np.ones((10,1),dtype='float64')
    #a = np.array([0,1],dtype = 'float64')
    #arr = np.tile(a,128)
    
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_weights,tf.keras.backend.cast_to_floatx(arr)))

    return 0.01 * loss

model = tf.keras.models.load_model('./original_3300_10.h5',custom_objects={'custom': custom})
#print(model.evaluate(X_test, Y_test))

import tensorflow_model_optimization as tfmot

quantize_scope = tfmot.quantization.keras.quantize_scope
quantize_model = tfmot.quantization.keras.quantize_model

with quantize_scope({'custom':custom}):
    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



X_train_subset = X_train[0:1000] # out of 60000
Y_train_subset = Y_train[0:1000]

q_aware_model.fit(X_train_subset, Y_train_subset,
                  batch_size=500, epochs=50, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(
    X_test, Y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   X_test, Y_test, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

q_aware_model.summary()

# モデルの保存
q_aware_model.save('./q_aware.h5')

# モデルの保存
#model.save('./simple_reg.h5')
 
#評価 & 評価結果出力
#print(model.evaluate(X_test, Y_test))



#電子透かし出力
layer = q_aware_model.layers[4]     
#print(q_aware_model.layers)
#print(layer.get_weights())

w = layer.get_weights()[1]
w = np.array(w)
np.set_printoptions(threshold=np.inf)


w = np.mean(w,axis=3)
w = np.reshape(w,[576,1])
x = np.random.rand(10,576)

x_weights = np.matmul(x,w)

# シグモイド関数の定義
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

x_weights = sigmoid(x_weights)

print(x_weights)


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

import numpy as np



# Create float TFLite model.
float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

# Measure sizes of models.
_, float_file = tempfile.mkstemp('.tflite')
_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))


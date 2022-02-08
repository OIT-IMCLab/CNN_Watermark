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
    w = tf.reshape(w,[576,1])

    x = tf.constant(np.random.rand(256,576),dtype=tf.float32)
    #print(x.numpy)
    #x = tf.fill((288,1), 1.0) #秘密鍵行列xの簡単な例として、0.5の値を持つ行列を作成
    x_weights = tf.matmul(x, w)  #行列の積の結果、100行かける1列の行列になる

    x_weights = tf.sigmoid(x_weights) #100個のデータをまとめて1個にしてから、シグモイド関数
    
    arr = np.ones((256,1),dtype='float64')
    
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_weights,tf.keras.backend.cast_to_floatx(arr)))

    return 0.02 * loss

model = tf.keras.models.load_model('./iwait_model/original_200_another_256.h5',custom_objects={'custom': custom})
#print(model.evaluate(X_test, Y_test))

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 50
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                               final_sparsity=0.8,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer=SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(X_train, Y_train,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   X_test, Y_test, verbose=0)

#評価 & 評価結果出力
print('model acc:',model.evaluate(X_test, Y_test))
print('Pruned test accuracy:', model_for_pruning_accuracy)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save('./iwait_model/pruning_200_256.h5')


"""
# モデルの保存
model.save('./simple_reg.h5')
 
#評価 & 評価結果出力
print(model.evaluate(X_test, Y_test))
 """
original_layer = model.layers[5]
original_w = original_layer.get_weights()[0]
original_w = np.array(original_w)
#print(original_w)

layer = model_for_export.layers[1]     #summaryよりInput->[0], Dense->[1]なのでmodel.layers[1]

w = layer.get_weights()[0]
w = np.array(w)
#print(w)
b = layer.get_weights()[1]
b = np.array(b)
#print('**Parameters shape**')
#print('w.shape', w.shape)
#print('b.shape', b.shape)
np.set_printoptions(threshold=np.inf)
#print(np.where(w > 0.9,1,0))
#print('b = ', b)

w = np.mean(w,axis=3)
w = np.reshape(w,[576,1])
x = np.random.rand(256,576)

#x = np.full((288,1), 1.0)
x_weights = np.matmul(x, w)

# シグモイド関数の定義
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

x_weights = sigmoid(x_weights)

x_weights = x_weights.flatten()
print(x_weights)


from tensorflow.python.keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train,x_test = x_train/255.0,x_test/255.0

from tensorflow.keras.utils import to_categorical
#第2引数の10は、今回のCIFAR10は10分類するため。
y_train,y_test = to_categorical(y_train,10),to_categorical(y_test,10)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense, Dropout, Activation, Flatten,InputLayer
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import random
import os
from tensorflow.keras import regularizers
import tempfile

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

def custom(weights):
    w = tf.reduce_mean(weights,axis=3)
    w = tf.reshape(w,[576,1])

    x = np.random.rand(256,576)
    #x /= x.sum(axis=1)[:,np.newaxis]
    #x = np.random.normal(0,1,(256,576))
    x = tf.constant(x,dtype=tf.float32)
    
    #print(x.numpy)
    #x = tf.fill((288,1), 1.0) #秘密鍵行列xの簡単な例として、0.5の値を持つ行列を作成
    x_weights = tf.matmul(x, w)  #行列の積の結果、100行かける1列の行列になる

    x_weights = tf.sigmoid(x_weights) #100個のデータをまとめて1個にしてから、シグモイド関数
    
    arr = np.ones((256,0),dtype='float32')
    c = np.arange(0,256).reshape(256, 1)
    condition = c % 2 == 0
    c[condition] = 1
    c[~condition] = 0
    #print(c)

    #arr = tf.constant(c, dtype='float32')
    
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_weights,tf.keras.backend.cast_to_floatx(arr)))

    return 0.02 * loss

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

params = {
    "padding": "same",
    "use_bias": True,
    "kernel_initializer": "he_normal",
}

model = Sequential()

model.add(layers.Conv2D(64, 3, 1, **params, input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
#model.add(layers.Conv2D(64, 3, 1, padding='same',use_bias=True))
model.add(layers.Conv2D(64, 3, 1, padding='same',use_bias=True,kernel_regularizer=custom))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(2, padding="same"))
model.add(layers.Conv2D(128, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(128, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(2, padding="same"))
model.add(layers.Conv2D(256, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(256, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(256, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(2, padding="same"))
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(2, padding="same"))
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(512, 3, 1, **params))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(2, padding="same"))
model.add(layers.Flatten())
model.add(layers.Dense(4096))
model.add(layers.Dense(4096))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

sgd = SGD(lr=0.1,momentum=0.9,nesterov=True)
model.compile(optimizer="SGD",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.25, epochs=50, batch_size=8, verbose=1)

print(model.evaluate(x_test,y_test))

model.save('./original_3300_256_all0.h5')

#print(model.layers)
layer = model.layers[3]
w = layer.get_weights()[0]
w = np.array(w)
print(w)
#print(w.shape)

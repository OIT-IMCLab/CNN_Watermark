import tensorflow as tf
import numpy as np

def custom(weights):
    w = tf.reduce_mean(weights,axis=3)
    w = tf.reshape(w,[288,1])

    #x = np.random.rand(10,36864)
    #x /= x.sum(axis=1)[:,np.newaxis]
    x = np.random.normal(0,1,(256,288))
    x = tf.constant(x,dtype=tf.float32)
    
    #print(x.numpy)
    #x = tf.fill((288,1), 1.0) #秘密鍵行列xの簡単な例として、0.5の値を持つ行列を作成
    x_weights = tf.matmul(x, w)  #行列の積の結果、100行かける1列の行列になる

    x_weights = tf.sigmoid(x_weights) #100個のデータをまとめて1個にしてから、シグモイド関数
    
    #arr = np.ones((256,1),dtype='float32')
    c = np.arange(0,256).reshape(256, 1)
    condition = c % 2 == 0
    c[condition] = 1
    c[~condition] = 0
    #print(c)

    arr = tf.constant(c, dtype='float32')
    
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x_weights,tf.keras.backend.cast_to_floatx(arr)))

    return 0.1 * loss




# 新しいモデルのインスタンスを作成
model = tf.keras.models.load_model('./original_200_256_all0.h5',custom_objects={'custom': custom})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
tflite_quant_model = converter.convert()
open("./quantize_200_256_all0.tflite", "wb").write(tflite_quant_model)

"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
"""

"""
converter=tf.lite.TFLiteConverter.from_keras_model("CIFAR-10-reg.h5")
#converter.post_training_quantize=True
tflite_quantized_model=converter.convert()
open("weights.tflite", "wb").write(tflite_quantized_model)
"""


 

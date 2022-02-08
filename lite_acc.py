import numpy as np
import tensorflow as tf

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


interpreter = tf.lite.Interpreter(model_path="./iwait_model/p-q_3300_256.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

i = 0
cnt = 0


while(i<10000):

    input_data = np.reshape(X_test[i], input_shape)

    interpreter.set_tensor(input_details[0]['index'], np.array(input_data, dtype=np.float32))

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    max_index = np.argmax(output_data)
    max_y = np.argmax(Y_test[i])
    #print(max_index)

    print(Y_test[i])
    if(max_y == max_index):
        cnt += 1
    i+=1

print(str(cnt) + "%")

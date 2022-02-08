import tensorflow as tf
import numpy as np
import random
import os

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


'''
Create interpreter, allocate tensors
'''
tflite_interpreter = tf.lite.Interpreter(model_path='./quantize_200_256_all0.tflite')
tflite_interpreter.allocate_tensors()

'''
Check input/output details
'''
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

#print("== Input details ==")
#print("name:", input_details[0]['name'])
#print("shape:", input_details[0]['shape'])
#print("type:", input_details[0]['dtype'])
#print("\n== Output details ==")
#print("name:", output_details[0]['name'])
#print("shape:", output_details[0]['shape'])
#print("type:", output_details[0]['dtype'])


'''
Run prediction (optional), input_array has input's shape and dtype
'''
input_shape = input_details[0]['shape']
input_array = np.array(np.random.random_sample(input_shape), dtype=np.float32)
tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
tflite_interpreter.invoke()
output_array = tflite_interpreter.get_tensor(output_details[0]['index'])

'''
This gives a list of dictionaries. 
'''
tensor_details = tflite_interpreter.get_tensor_details()

for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    scales = dict['quantization_parameters']['scales']
    zero_points = dict['quantization_parameters']['zero_points']
    tensor = tflite_interpreter.tensor(i)()


    print(i, type,  scales.shape, zero_points.shape, tensor.shape)

w = tflite_interpreter.tensor(11)() #11or22
print(w)
np.set_printoptions(precision=25)
w = w / 100.0
w = np.mean(w,axis=3)
w = np.reshape(w,[288,1])
#x = np.full((288,1), 1.0)
x = np.random.rand(256,288)
#x = np.random.normal(0,1,(256,576))
#x /= x.sum(axis=1)[:,np.newaxis]

x_weights = np.matmul(x, w)

# シグモイド関数の定義
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

x_weights = sigmoid(x_weights)
x_weights = x_weights.flatten()
print(x_weights)

'''
See note below
'''

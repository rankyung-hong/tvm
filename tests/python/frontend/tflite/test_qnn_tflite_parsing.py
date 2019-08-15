from keras.preprocessing import image
import tvm.relay as relay
import tvm
from tvm.contrib import graph_runtime as runtime
import numpy as np
import tflite.Model
import time

tflite_model_file = "test_models/inception_v1_224_quant.tflite"
# tflite_model_file = "test_models/inception_v2_224_quant.tflite"
# tflite_model_file = "test_models/inception_v3_299_quant.tflite"
# tflite_model_file = "test_models/inception_v4_299_quant.tflite"
# tflite_model_file = "test_models/mobilenet_v2_1.0_224_quant.tflite"
# tflite_model_file = "test_models/inception_v3_299.tflite"
# tflite_model_file = "test_models/inception_v4_299.tflite"
# tflite_model_file = "test_models/mobilenet_v2_1.0_224.tflite"

input_shape = (1, 224, 224, 3)
# input_shape = (1, 299, 299, 3)

dtype = 'uint8'
# dtype = 'float32'

input_tensor = "input"
ctx = tvm.cpu(0)
target = "llvm -mcpu=core-avx2"

img_paths = []
img_paths.append('test_images/banana_954.jpg')
img_paths.append('test_images/corn_987.jpg')
img_paths.append('test_images/lemon_951.jpg')
img_paths.append('test_images/strawberry_949.jpg')
img_paths.append('test_images/tench_000.jpg')
img_paths.append('test_images/toilet_tissue_999.jpg')

current_milli_time = lambda: int(round(time.time() * 1000))

def get_data(img_paths, input_shape, dtype):
    data = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
        x = image.img_to_array(img, dtype=dtype)
        x = np.expand_dims(x, axis=0)
        if dtype == "float32":
            x[:, :, :, 0] = 2.0 / 255.0 * x[:, :, :, 0] - 1
            x[:, :, :, 1] = 2.0 / 255.0 * x[:, :, :, 1] - 1
            x[:, :, :, 2] = 2.0 / 255.0 * x[:, :, :, 2] - 1
        data.append(x)
    return data

preprocessed_data = get_data(img_paths, input_shape, dtype)
shape_dict = {input_tensor: input_shape}

tflite_model_buf = open(tflite_model_file, "rb").read()
tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

answer = dict()
with open("test_images/answer.txt", "r") as f:
    for line in f:
        blocks = line.split(":")
        text = blocks[1].split("\n")
        answer[int(blocks[0])] = text[0]

t1 = current_milli_time()
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: dtype})
print(mod.astext(show_meta_data=False))
t2 = current_milli_time()
print("Parse TFLite model: {} ms".format(t2 - t1))
t1 = current_milli_time()

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)
t2 = current_milli_time()
print("Build quantized model: {} ms".format(t2 - t1))

module = runtime.create(graph, lib, tvm.cpu())
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=30, repeat=3)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))

for idx, data in enumerate(preprocessed_data):
    module.set_input(input_tensor, tvm.nd.array(data))
    module.set_input(**params)
    module.run()
    tvm_output = module.get_output(0).asnumpy()
    predictions = np.squeeze(tvm_output)
    print("\n")
    print("file name: {}".format(img_paths[idx]))
    top3 = predictions.argsort()[-3:][::-1] - 1
    print("*** top 3 ***")
    print("max value: {}".format(np.max(predictions)))
    print(str(top3[0]) + ": " + answer[top3[0]])
    print(str(top3[1]) + ": " + answer[top3[1]])
    print(str(top3[2]) + ": " + answer[top3[2]])




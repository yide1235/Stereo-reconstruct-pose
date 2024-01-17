

import time
# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the input image.
# image_path = 'tmp.jpg'
def run_thunder(image_path):
    model_path = "./THUNDER/model.tflite"
    # model_path = "./THUNDER/model_light.tflite"
    movenet_threshold=0.45

    model_size=0
    if model_path=="./THUNDER/model_light.tflite":
      model_size=192
    elif model_path=="./THUNDER/model.tflite":
      model_size=256

    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    dim_prior=image.shape
    if dim_prior[1]>dim_prior[2]:
        y_shift=0
        x_shift=(model_size-(dim_prior[2]/(dim_prior[1]/model_size)))//2
        scale=dim_prior[1]/model_size
    else:
        y_shift=(model_size-(dim_prior[1]/(dim_prior[2]/model_size)))//2
        x_shift=0
        scale=dim_prior[2]/model_size

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, model_size, model_size), dtype=tf.uint8)

    # Save the image using a suitable library (PIL, OpenCV, etc.)
    # Here, we'll use PIL (Python Imaging Library):
    output_image_path = 'model_input.jpg'
    image_array = tf.squeeze(image, axis=0).numpy().astype(np.uint8)
    result_image = Image.fromarray(image_array)
    # result_image.save(output_image_path)


    # # Download the model from TF Hub.
    # model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

    # movenet = model.signatures['serving_default']
    # # Run model inference.
    # outputs = movenet(image)

    # model_path = "./THUNDER/model.tflite"
    
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    start_time = time.time()

    interpreter.invoke()
    end_time = time.time()

    duration = end_time - start_time
    # print(f"Time taken for 'invoke': {duration} seconds")
    
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    # print(keypoints)

    # Compute keypoints location in image
    # keypoints = outputs['output_0']
    conf_sum=0

    points=[]
    for point in keypoints[0][0]:
        # print(point)
        #points in this format: [0.14852381 0.57402444 0.5017696 ]
        if(point[2]>movenet_threshold):
          # print('----------------')
          # points.append([(-x_shift+(point[1]*256))*scale,(-y_shift+(point[0]*256))*scale])
          points.append([(-x_shift+(point[1]*model_size))*scale,(-y_shift+(point[0]*model_size))*scale])
        else:
          points.append([0,0])

    # print(keypoints)
    # print(points)
    # print(keypoints)
    # print(points)
    min_conf=1
    for i in keypoints[0][0][5:17]:
      if i[2]<min_conf:
 
        min_conf=i[2]
      # conf_sum+=point[2]
    
    # print(keypoints,min_conf)

        
    # return [points], conf_sum
    return [points], min_conf






















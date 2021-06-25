#This app consume an API exposed by TensorFlow Serving to scoring an image through an Object Detector ML model
#by Augusto Alves Furtado

import json
import numpy as np
from PIL import Image
import requests
import tensorflow as tf 

url = 'http://localhost:8501/v1/models/ssd_resnet50:predict'
image_filename = 'C:\\workspace\\TensorFlow\\workspace\\training_demo\\dataset_to_validate\\tmp\\table_pen.jpg'

#instantiate an image object
image = Image.open(image_filename)
#load image into a numpy array
image_np = np.array(image)
# The input needs to be a tensor
image_tf = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with "tf.newaxis"
#Note: otherwise erro similar to "Input shape axis 0 must equal 1, got shape [1200,1600,3]"
image_tf = image_tf[tf.newaxis, ...]

#not necessary (the instruction right below) to make the image as 3D, the instruction above works as expected by the ML model
#image_tf = image_tf.reshape(inputs, [1, -1, 1])  # Make it 3D		  

#serialize the content as json
content = json.dumps({"signature_name": "serving_default", "instances": image_tf.numpy().tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(url, data=content, headers=headers)

#Take only index [0] from the response to remove the batch dimension
#Note: There are other dimensions with predictions, we are only interested in the first dimension.
predictions = json.loads(json_response.text)['predictions'][0]

print('response status: ', json_response, '\n')
print('detection classes: \n', predictions['detection_classes']) # object label in the image
print('detection scores: \n', predictions['detection_scores']) # object scores in the image
print('detection boxes: \n', predictions['detection_boxes']) # object box position in the image

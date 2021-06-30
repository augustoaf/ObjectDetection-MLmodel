#by Augusto Alves Furtado

import json
import numpy as np
from PIL import Image
import requests
import tensorflow as tf 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import matplotlib.pyplot as plt
import time
import paho.mqtt.client as mqtt

URL_API = 'http://192.168.86.42:8501/v1/models/ssd_resnet50:predict'
LABEL_MAP_WITH_PATH = '/app/mscoco_label_map.pbtxt'
MINIMUM_SCORE_TO_PLOT = .4
LABEL_MAP = None
BROKER_HOST = "192.168.86.42"
BROKER_PORT = 1883
MQTT_TOPIC = "house/pictures"
IMAGE_REQUESTS_QUEUE = []
PICTURES_PATH = '/app/pictures/'

def add_image_request_on_queue(message_converted):
    global IMAGE_REQUESTS_QUEUE
    IMAGE_REQUESTS_QUEUE.append(message_converted)

# mqtt callback method to act when the connection is closed by the broker or by the client
def on_disconnect(client, userdata, rc):
    print("client disconnected status: ", rc)

# mqtt callback for connection
def on_connect(client, userdata, flags, rc):
    print("client connected status: ", rc)
    client.subscribe(MQTT_TOPIC)

# callback method to receive the message when published on the topic this client has subscribed
def on_message(client, userdata, message):
    message_converted = str(message.payload.decode("utf-8"))
    print("message received on topic ", message.topic, ": ", message_converted)
    add_image_request_on_queue(message_converted)

# callback method for log
def on_log(client, userdata, level, buf):
    print("log: ",buf)

def instantiate_mqtt_client():
    client = None
    try:
        client = mqtt.Client()
        client.connect(BROKER_HOST, port=BROKER_PORT, keepalive=60)

        #attach callback functions
        client.on_message=on_message
        client.on_disconnect = on_disconnect
        client.on_connect = on_connect
        client.on_log=on_log

        client.loop_start()
    except Exception as e:
        error = f"MQTT Client: Logging exception as repr: {e!r}"
        raise Exception(error)
        #print(error)
            
    return client

def get_label_map():
    #applying LABEL_MAP a singleton pattern
    #in order to modify a global variable in Python it is necessary to declare "global <variable name>" first
    global LABEL_MAP
    if LABEL_MAP is None:
        LABEL_MAP = label_map_util.create_category_index_from_labelmap(LABEL_MAP_WITH_PATH, use_display_name=True) 
        print('\nlabel map instantiated\n')
    return LABEL_MAP

def get_labeled_image_filename(original_image_filename):
    new_image_filename = original_image_filename[0:len(original_image_filename)-4] + '_detection' + original_image_filename[len(original_image_filename)-4:4] 
    return new_image_filename

def plot_detections_and_save_new_image(image_detections, label_map, original_image_numpy_array, path_filename_to_save, score_threshold):
    
    #making a copy because the visualize_boxes_and_labels_on_image_array function override the input image
    image_numpy_array = original_image_numpy_array.copy()

    #convert arrays to numpy array - classes must have values as int
    image_detections['detection_boxes'] = np.array(image_detections['detection_boxes'])
    image_detections['detection_classes'] = np.array(image_detections['detection_classes']).astype(np.int32)
    image_detections['detection_scores'] = np.array(image_detections['detection_scores'])

    #This function groups boxes with label for each detection and overlays these on the input image and return it.
    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_numpy_array,
        image_detections['detection_boxes'], # object box position in the image
        image_detections['detection_classes'], # object label in the image
        image_detections['detection_scores'], # object scores in the image
        label_map,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=score_threshold, #accuracy - values between 1.0 (high accuracy) and 0.0 (lowest accuracy)
        agnostic_mode=False)
	
    #save labeled image
    plt.figure() # Create a new figure, or activate an existing figure. 
    plt.imshow(image_np_with_detections) # Display data as an image
    plt.savefig(
        path_filename_to_save, 
	    facecolor='w', 
	    edgecolor='w', 
	    orientation='portrait', 
	    transparent=False, 
	    pad_inches=0.1)
    plt.close()

def print_detections(image_detections, label_map, score_threshold):
    #print objects detected in the image accoring the minumum score threshold provided
    print('Objects detected with score >= ', score_threshold, end='\n')
    index = 0
    scores = image_detections['detection_scores']
    for score in scores:
        if score >= score_threshold:
            print('label: ', label_map.get(image_detections['detection_classes'][index]), ' score: ', image_detections['detection_scores'][index], end='\n')
        index = index + 1

def detect_objects(image_filename):
    #ML model label mappings
    label_map = get_label_map()

    try:
        #instantiate an image object
        image = Image.open(image_filename)
    except Exception as e:
        raise FileNotFoundError('error to open the image file: ' + image_filename + '\n')
    print('Object Detection started for : ', image_filename, end ='\n')

    #load image into a numpy array
    image_np = np.array(image)
    #release resource
    image.close()
    # The input needs to be a tensor
    image_tf = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with "tf.newaxis"
    #Note: otherwise erro similar to "Input shape axis 0 must equal 1, got shape [1200,1600,3]"
    image_tf = image_tf[tf.newaxis, ...]

    #not necessary (the instruction right below) to make the image as 3D, the instruction above works as expected by the ML model
    #image_tf = image_tf.reshape(inputs, [1, -1, 1])  # Make it 3D		  

    #serialize the content as json
    content = json.dumps({"signature_name": "serving_default", "instances": image_tf.numpy().tolist()})
    #consume TensorFlow Serving REST API
    json_response = requests.post(URL_API, data=content, headers={"content-type": "application/json"})

    #Get only index[0] from the response to remove the batch dimension
    #Note: There are other dimensions with predictions, we are only interested in the first dimension.
    alldimensions_predictions = json.loads(json_response.text)['predictions']
    predictions = alldimensions_predictions[0]    

    labeled_image_filename_to_save = get_labeled_image_filename(image_filename)
    print('API response status: ', json_response, '\n')
    plot_detections_and_save_new_image(predictions, label_map, image_np, labeled_image_filename_to_save, MINIMUM_SCORE_TO_PLOT)
    print('\n labeled image saved: ', labeled_image_filename_to_save)
    print_detections(predictions, label_map, MINIMUM_SCORE_TO_PLOT)

def main():

    client = None
    try:
        client = instantiate_mqtt_client()

        while True:            
            if len(IMAGE_REQUESTS_QUEUE) == 0:
                time.sleep(5)
                print("listening topic ", MQTT_TOPIC, ' for image object detection ...', end='\n')
            else:
                #get the next element in the queue - FIFO
                image_request = IMAGE_REQUESTS_QUEUE.pop(0)
                try:
                    detect_objects(PICTURES_PATH + image_request)
                except FileNotFoundError as e:
                    print(e)
    except KeyboardInterrupt:
        if not client is None:
            client.loop_stop()
            client.disconnect()
        print('App stopped')
    except Exception as e:
        error = f"Error: {e!r}"
        print(error)
        if not client is None:
            client.loop_stop()
            client.disconnect()

if __name__ == '__main__':
    print ( "Press Ctrl-C to exit" )
    main()
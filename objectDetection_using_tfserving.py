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

URL_API = 'http://192.168.86.42:8501/v1/models/ssd_resnet50:predict'
LABEL_MAP_WITH_PATH = 'C:\\workspace\\TensorFlow\\workspace\\training_demo\\annotations\\COCO_labels\\mscoco_label_map.pbtxt'
IMAGE_FILENAME_WITH_PATH = 'C:\\workspace\\TensorFlow\\workspace\\training_demo\\dataset_to_validate\\tmp\\table_pen.jpg'
MINIMUM_SCORE_TO_PLOT = .4

def get_label_map():
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_WITH_PATH, use_display_name=True) 
    return category_index

def get_labeled_image_filename(original_image_filename_with_path):
    image_filename = os.path.basename(original_image_filename_with_path)
    image_path = original_image_filename_with_path[0:len(original_image_filename_with_path)-len(image_filename)]
    new_image_filename = image_filename[0:len(image_filename)-4] + '_detection' + image_filename[len(image_filename)-4:4] 
    return image_path + new_image_filename

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

def print_detections(image_detections, label_map, score_threshold):
    #print objects detected in the image accoring the minumum score threshold provided
    print('Objects detected with score >= ', score_threshold, end='\n')
    index = 0
    scores = image_detections['detection_scores']
    for score in scores:
        if score >= score_threshold:
            print('label: ', label_map.get(image_detections['detection_classes'][index]), ' score: ', image_detections['detection_scores'][index], end='\n')
        index = index + 1

def main():
    #ML model label mappings
    label_map = get_label_map()

    #instantiate an image object
    image = Image.open(IMAGE_FILENAME_WITH_PATH)
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
    #consume TensorFlow Serving REST API
    json_response = requests.post(URL_API, data=content, headers={"content-type": "application/json"})

    #Get only index[0] from the response to remove the batch dimension
    #Note: There are other dimensions with predictions, we are only interested in the first dimension.
    alldimensions_predictions = json.loads(json_response.text)['predictions']
    predictions = alldimensions_predictions[0]    

    labeled_image_filename_to_save = get_labeled_image_filename(IMAGE_FILENAME_WITH_PATH)
    print('API response status: ', json_response, '\n')
    plot_detections_and_save_new_image(predictions, label_map, image_np, labeled_image_filename_to_save, MINIMUM_SCORE_TO_PLOT)
    print('\n labeled image saved: ', labeled_image_filename_to_save)
    print_detections(predictions, label_map, MINIMUM_SCORE_TO_PLOT)

if __name__ == '__main__':
    print ( "Press Ctrl-C to exit" )
    main()
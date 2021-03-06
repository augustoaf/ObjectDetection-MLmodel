#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model - customized by Augusto
==========================================================================
"""

# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
##gpus = tf.config.experimental.list_physical_devices('GPU')
##for gpu in gpus:
##    tf.config.experimental.set_memory_growth(gpu, True)

ROOT_FOLDER = os.getcwd() #this will get the current path where the python program is running

def get_images():
	base_images_path = ROOT_FOLDER + '\\dataset_to_validate\\tmp\\'
	filenames = ['table_pen.jpg', 'pen21.jpg', 'spoon.jpg', 'blank_page.jpg', 'black_page.jpg']  #provide an array of images
	image_paths = []
	for filename in filenames:
		image_path = pathlib.Path(base_images_path + filename)
		image_paths.append(image_path)
	return image_paths

IMAGE_PATHS = get_images()
PATH_TO_MODEL_DIR = ROOT_FOLDER + '\\pre-trained-models\\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8' #pre-trained ML model available by TensorFlow Model Garden
PATH_TO_LABELS = ROOT_FOLDER + '\\annotations\\COCO_labels\\mscoco_label_map.pbtxt' #labels mapping based on COCO 2017
#PATH_TO_MODEL_DIR = ROOT_FOLDER + '\\exported_models\\my_ssd_resnet50_v1_fpn' #custom ML model
#PATH_TO_LABELS = ROOT_FOLDER + '\\annotations\\label_map.pbtxt' #custom labels mapping

# Load the model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "\\saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# Run the detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print('\n Running inference for {}... '.format(image_path), end='\n')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    #perform the detections over the ML model
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
	
    score_threshold = 0.4

	#This function groups boxes with label for each detection and overlays these on the input image and return it.
    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'], # object box position in the image
          detections['detection_classes'], # object label in the image
          detections['detection_scores'], # object scores in the image
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=score_threshold, #accuracy - values between 1.0 (high accuracy) and 0.0 (lowest accuracy)
          agnostic_mode=False)
	
	#save labeled image
    image_path_str = str(image_path)
    labeled_image_filename = image_path_str[0:len(image_path_str)-4] + '_detection' + image_path_str[len(image_path_str)-4:4]
    plt.figure() # Create a new figure, or activate an existing figure. 
    plt.imshow(image_np_with_detections) # Display data as an image
    plt.savefig(
          labeled_image_filename, 
		  facecolor='w', 
		  edgecolor='w', 
		  orientation='portrait', 
		  transparent=False, 
		  pad_inches=0.1)
	
	#print labeled image filenames and the objects detected
    print('labeled image', labeled_image_filename, end='\n')
    index = 0
    scores = detections['detection_scores']
    for score in scores:
        if score >= score_threshold:
            print('label id: ', detections['detection_classes'][index], ' score: ', detections['detection_scores'][index], end='\n')
        index = index + 1
    	
    print('Done')

plt.show() #it is suppose to open a graphical user interface on your OS to show the image


------------------------------------------------------------------------------------------------------  
App: objectDetection_using_tfserving_vMQTTandDocker  
  
This app runs under a docker container and consume an API exposed by TensorFlow Serving (running through a Docker container) to scoring an image through an Object Detector ML model (ssd_resnet50_v1_fpn_640x640_coco17_tpu-8) and save a labeled image  
Trigger: by MQTT topic where it receives an image filename (without path) to start the Object Detection process  
  
Assumptions:  
  -ML model scoring method available through REST API: http://localhost:8501/v1/models/ssd_resnet50:predict  
  -MQTT Broker running in order to the App read topic messages from house/pictures - text must be a picture filename  
  -Pictures are mapped from a host folder to a docker container folder where it will load the picture and also save a new labeled picture  
  
------------------------------------------------------------------------------------------------------  
App: objectDetection_using_tfserving_vMQTT.py  
  
This app consume an API exposed by TensorFlow Serving (running through a Docker container) to scoring an image through an Object   Detector ML model (ssd_resnet50_v1_fpn_640x640_coco17_tpu-8) and save a labeled image  
Trigger: by MQTT topic where it receives an image filename (with path) to start the Object Detection process  
  
Assumptions:  
  -ML model scoring method available through REST API: http://localhost:8501/v1/models/ssd_resnet50:predict  
  -MQTT Broker running in order to the App read topic messages from house/pictures - text must be a picture path+filename  
  -Picture and Object Label Mapping available in the host  
  
------------------------------------------------------------------------------------------------------  
App: objectDetection_using_tfserving.py  
  
This app consume an API exposed by TensorFlow Serving (running through a Docker container) to scoring an image through an Object  Detector ML model (ssd_resnet50_v1_fpn_640x640_coco17_tpu-8) and save a labeled image  
Trigger: by a hardcoded image filename set in the App code  
  
Assumptions:  
  -ML model scoring method available through REST API: http://localhost:8501/v1/models/ssd_resnet50:predict  
  -Picture and Object Label Mapping available in the host  
  
------------------------------------------------------------------------------------------------------  
Apps: plot_object_detection_checkpoint-byAugusto and plot_object_detection_saved_model-byAugusto  
  
Apps using TensorFlow and Object Detection API libs.  
Both apps uses a local ML model to detect objects based on an array of images and save the labeled images and also print out on the console the objects detected.  
  
Notes:  
-The ML model used are provided by TensorFlow Model Garden and they were trained using COCO dataset.  
-The apps were downloaded from TensorFlow website and customized by myself to work with local ML model.  
-There are two versions of the App, one to work with "saved model" and another to work with "model checkpoint".  
  
Requirements:  
ML model based on COCO dataset and Labels Mapping  
The app must run on the "workspace/training_demo" folder  
  
Explanation about the folder structure:  
  
workspace: your space to work on the models  
workspace/training_demo: it is your training folder, it is recommended to create a new structure for every new different dataset to be trained.  
workspace/training_demo/annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.  
workspace/training_demo/exported-models: This folder will be used to store exported versions of our trained model(s).  
workspace/training_demo/images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.  
workspace/training_demo/images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.  
workspace/training_demo/images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.  
workspace/training_demo/models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.  
workspace/training_demo/pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.  
workspace/training_demo/scripts: scripts to automate tasks  
-----------------------------------------------------------------------------------------------------  


======================================================================================================

Create Docker TensorFlow Serving with your ML model embedded

Pull a TensorFlow Serving Docker image:
docker pull tensorflow/serving

Run TensorFlow Serving Docker Container as a daemon (detached of your current shell prompt):
docker run -d --name <daemon name> tensorflow/serving
Note: Give a name to your daemon.
E.g.: docker run -d --name serving_base tensorflow/serving

Copy saved ML model to TensorFlow Serving Docker Container model folder
docker cp models/<my model> <daemon name>:/models/<my model>
E.g.: docker cp ssd_resnet50/ serving_base:/models/
Note: The folder structure expected by the TensorFlow serving is /models/<your base model folder>/<version number>/, so the content copied (ssd_resnet50 folder) into the models folder has this structure: /ssd_resnet50/1/

Save the container based on the daemon container
docker commit --change "ENV MODEL_NAME <model name>" <daemon name> <my container>
Note: Provide a model name, daemon name and a Docker container name to be saved
E.g.: docker commit --change "ENV MODEL_NAME ssd_resnet50" serving_base tensorflow/myserving

Now you can run your new Docker container with a ML model already embedded 
docker run -t --rm -p 8501:8501 <docker container name> &
E.g.: docker run -t --rm -p 8501:8501 tensorflow/myserving &

======================================================================================================

Find below an example of object detection with accuracy above 0.4   
ML model used: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
Images: table_detection.png (labeled image) / table.jpg (original image)  
  
array of labels identified:  
  
 classes: [74 73 76 31 62 62 44 78 47 62 73 33 73 73 73 73 84 74 84 62 77 76 72 27  
 47 73 76 74 76 72 62 73 79 84 44 62 63 72 77 73 90 62 84 74 47 15 44 67  
 67 44 76 84 80 44 76 51 75 62 73 37 84 67 62 53 73 31 62 47 67 86  1 67  
 74 77 86 80 76 76 62 72 47 15 62 15 50 74 76 84 77 76 73 84 67 84 62 73  
 77 67 62 44]  
  
array of scores for each label (label name commented for the ones with accuracy above 0.4):   
  
scores: [0.7393426 (mouse)  0.6141983 (laptop)  0.59453744 (keyboard) 0.52625954 (handbag) 0.51244193 (chair) 0.46622196 (chair)  
 0.43818137 (bottle) 0.36671653 0.34830058 0.32359064 0.32174313 0.32134733  
 0.31929985 0.31570804 0.30758217 0.27697784 0.27041996 0.25595847  
 0.24610159 0.22949332 0.22720191 0.22650528 0.21397948 0.2064358  
 0.19932109 0.19386214 0.19249469 0.18330607 0.17892295 0.17837319  
 0.17801312 0.17061177 0.16420817 0.16088307 0.15910786 0.15789756  
 0.15692061 0.15560845 0.15152559 0.14944038 0.14884499 0.14574462  
 0.1426163  0.13740173 0.13728738 0.13621011 0.13491192 0.1345208  
 0.13262516 0.12968415 0.1291227  0.12871593 0.12862009 0.1264283  
 0.1262905  0.12575924 0.12569928 0.12555441 0.12546256 0.12421966  
 0.1229355  0.1185351  0.11850056 0.1184974  0.11705273 0.11677817  
 0.11295027 0.11291689 0.11107299 0.11043519 0.10917664 0.10892558  
 0.10767537 0.10610271 0.1054095  0.10509074 0.10492161 0.10483754  
 0.1048356  0.10324192 0.10305911 0.10271439 0.10262623 0.10205591  
 0.1017791  0.10169038 0.10121891 0.09998691 0.09961921 0.09952316  
 0.0994617  0.09892714 0.09801289 0.09706995 0.09673703 0.09592593  
 0.09516573 0.09499881 0.09395179 0.09320053]  
  
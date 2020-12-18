import sys, os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input as pp_xception
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as pp_vgg16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as pp_vgg19
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as pp_inceptionv3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as pp_mobilenetv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as pp_resnet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as pp_resnet50v2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input as pp_resnet101v2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input as pp_resnet152v2
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as pp_densenet121
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input as pp_densenet169
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input as pp_densenet201
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input as pp_nasnetmobile
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as pp_efficientnetb0
from keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from memory_profiler import profile
import cv2
import argparse

models = [
    [Xception, 'xception', pp_xception],
    [VGG16, 'vgg16', pp_vgg16],
    [VGG19, 'vgg19', pp_vgg19],
    [InceptionV3, 'inceptionv3', pp_inceptionv3],
    [MobileNetV2, 'mobilenetv2', pp_mobilenetv2],
    [ResNet50V2, 'resnet50v2', pp_resnet50v2],
    [ResNet101V2, 'resnet101v2', pp_resnet101v2],
    [ResNet152V2, 'resnet152v2', pp_resnet152v2],
    [DenseNet121, 'densenet121', pp_densenet121],
    [DenseNet169, 'densenet169', pp_densenet169],
    [DenseNet201, 'densenet201', pp_densenet201],
    [NASNetMobile, 'nasnetmobile', pp_nasnetmobile],
    [EfficientNetB0, 'efficientnetb0', pp_efficientnetb0]
]

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.Session(config=config)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

@profile
def memoryPerImage(i):

	modelChosen = models[i][0]
	modelName = models[i][1]
	preprocess = models[i][2]

	fileName  = '.keras/models/' + modelName +'.h5'
	model = modelChosen(weights = fileName)


	if modelName not in ['xception', 'inceptionv3']:
		inputShape = (224, 224)
	else:
		inputShape = (299, 299)
		
	imageLoc = 'images/cat.jpeg'
	image = load_img(imageLoc, target_size=inputShape)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess(image)

	print("Classifying image with {}".format(modelName))
	preds = model.predict(image)
	P = imagenet_utils.decode_predictions(preds)
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
	
	K.clear_session()


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, type=int)
args = vars(ap.parse_args())


memoryPerImage(args["model"])



















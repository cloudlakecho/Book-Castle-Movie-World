
# main.py
# Cloud Cho - Book Segmentation
#
# May 6, 2018 ~
#
# Step
# (1) Book segmentation
# (2) Title collection
#
# To do:
#
# Error:
#   OpenCV version collision at faster_rcnn/simple_parser.py (essential)
#   RoiPoolingConv function should be added in faster_rcnn
#   RoiPoolingConv input variable cd
#
# Source:
#   1st Trial
#       https://github.com/FraPochetti/ImageTextRecognition
#   2nd Trial
#
# Work? - no

from __future__ import division
# from data import OcrData
# from cifar import Cifar
# from userimageski import UserData

import ipdb
import random
import pprint
import sys
import time
import numpy as np
import math
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
# from faster_rcnn import config, data_generators
# from faster_rcnn import losses as losses
# import faster_rcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

# from faster_rcnn import simple_parser, vgg_sixteen
from faster_rcnn import vgg_sixteen

# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# 1st Trial
def detection_model():
    ################################################################
    # 1- GENERATE MODEL TO PREDICT WHETHER AN OBJECT CONTAINS TEXT OR NOT
    ################################################################

    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA
    data = OcrData('/home/francesco/Dropbox/DSR/OCR/ocr-config.py')

    # GENERATES A UNIQUE DATA SET MERGING NON-TEXT WITH TEXT IMAGES
    data.merge_with_cifar()

    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
    data.perform_grid_search_cv('linearsvc-hog')

    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
    data.generate_best_hog_model()

    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
    data.evaluate('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain2-90.pickle')

def extraion_model():
    ###################################################################
    # 2- GENERATE MODEL TO CLASSIFY SINGLE CHARACTERS
    ###################################################################

    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA
    data = OcrData('/home/francesco/Dropbox/DSR/OCR/ocr-config.py')

    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
    data.perform_grid_search_cv('linearsvc-hog')

    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
    data.generate_best_hog_model()

    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
    data.evaluate('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain36-90.pickle')


def test_model():
    ##### the following code includes all the steps to get from a raw image to a prediction.
    ##### the working code is the uncommented one.
    ##### the two pickle models which are passed as argument to the select_text_among_candidates
    ##### and classify_text methods are obviously the result of a previously implemented pipeline.
    ##### just for the purpose of clearness below the code is provided.
    ##### I want to emphasize that the commented code is the one necessary to get the models trained.

    # creates instance of class and loads image
    user = UserData('lao.jpg')
    # plots preprocessed imae
    user.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = user.get_text_candidates()
    # plots objects detected
    user.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    maybe_text = user.select_text_among_candidates('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain2-90.pickle')
    # plots objects after text detection
    user.plot_to_check(maybe_text, 'Objects Containing Text Detected')
    # classifies single characters
    classified = user.classify_text('/media/francesco/Francesco/CharacterProject/linearsvc-hog-fulltrain36-90.pickle')
    # plots letters after classification
    user.plot_to_check(classified, 'Single Character Recognition')
    # plots the realigned text
    user.realign_text()

# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# 2nd Trial
def train_model(dataset, roi):
    # (1) Prepare data and label
    input_shape_img = (None, None, 3)  # width, height, depth
    # number of roi, top left, bottom right coordinate
    roi_shape = (None, 4)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=roi_shape)

    # Error spot
    # all_imgs, classes_count, class_mapping = simple_parser.get_data(train_path)
    classes = ['human', 'car']
    # simple_parser.py shoud be fixed
    classes_count = {classes[0]: 10, classes[1]: 10}

    # define the base network
    shared_layers = vgg_sixteen.nn_base(img_input, trainable=True)

    # (2) Neural Network defining
    # define the RPN, built on the base layers
    anchor_box_scales = [128, 256, 512]
    anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)],
                        [2./math.sqrt(2), 1./math.sqrt(2)]]
    num_rois = 32
    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
    rpn = vgg_sixteen.rpn(shared_layers, num_anchors)

    # ipdb.set_trace()

    classifier = vgg_sixteen.classifier(shared_layers, roi_input, num_rois,
                 nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)




    # Error:
    # "Layer model_3 was called with an input that isn't a symbolic tensor.
    # Received type: <class 'numpy.ndarray'>.
    #
    try:
        # This is a model that holds both the RPN and the classifier, used
        # to load/save weights for the models
        # dataset: image, labels
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)([dataset[0], roi])
    except Exception as e:
        print (e.args)
        ipdb.set_trace(context=10)
    else:
        print ('Training completed')

# -----
#
# -----
def main():
    # # 1st Trial
    # detection_model()
    # extraction_model()
    #
    # test_model()

    # 2nd Trial
    width = 12
    height = 12
    depth = 3
    dataset_size = 10
    roi_shape = (1, 4)
    data = np.random.randint(0, 2, (width, height, depth))
    label =  np.random.randint(0, dataset_size, (dataset_size))
    # ipdb.set_trace()
    roi = np.random.randint(0, np.maximum(width, height), roi_shape)

    print(data.shape, label.shape)

    dataset = [data, label]
    print(type(dataset))
    train_model(dataset, roi)

if __name__ == '__main__':
    main()

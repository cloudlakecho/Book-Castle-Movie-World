
# main.py
# Cloud Cho - Book Segmentation
#
# May 6, 2018 ~
#
# Step
# (1) Book segmentation
# (2) Title collection
#
# Source:
#   1st Trial
#   https://github.com/FraPochetti/ImageTextRecognition
#   2nd Trial
#
# Work? - no

from data import OcrData
from cifar import Cifar
from userimageski import UserData

from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

from tool import vgg_sixteen

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
def train_model(dataset):
    # (1) Prepare data and label
    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))
    # define the base network
    shared_layers = vgg_sixteen.nn_base(img_input, trainable=True)

    # (2) Neural Network defining
    # define the RPN, built on the base layers
    anchor_box_scales = [128, 256, 512]
    num_anchors = len(anchor_box_scales) * len(C.anchor_box_ratios)
    anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)],
                        [2./math.sqrt(2), 1./math.sqrt(2)]]
    rpn = vgg_sixteen.rpn(shared_layers, num_anchors)

# -----
#
# -----
def main():
    # 1st Trial
    detection_model()
    extraction_model()

    test_model()

    # 2nd Trial

if __name__ == '__main__':
    main()

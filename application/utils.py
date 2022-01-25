from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle 


def dog(dog_names):
    with open("model/dog_names", "rb") as fp:
        dog_names = pickle.load(fp)

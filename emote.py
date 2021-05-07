# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:01:14 2021

@author: jerry
"""
import json
import keras
from keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import random
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
JSON_PATH="data.json"
dataset_path="Dataset"
SAMPLES_TO_CONSIDER= 44100 #1sec
def prepare_dataset(dataset_path, json_path, n_mfcc=13, length=512, n_fft=2048):
    data={
          "mappings":[],
          "labels":[],
          "MFCCs": [],
          "files":[]

          }
    #loop through all the sub-dirs

    for i,(dirpath,dirnames,filenames) in enumerate (os.walk(dataset_path)):
        #ensure that we are not at the dataset level.
        if dirpath is not dataset_path:

              category=dirpath.split("/")[-1]
              data["mappings"].append(category)
              print(f"processing {category}")
              #extract MFCCs
              for f in filenames:

                  #get the filepath
                  file_path=os.path.join(dirpath, f)
                  #load the audio file
                  signal, sr=librosa.load(file_path)
                  #ensure the audio file is at least 1sec
                  if len(signal)>=SAMPLES_TO_CONSIDER:
                      #enforce 1sec long signal
                      signal=signal[:SAMPLES_TO_CONSIDER]
                      #extract the MFCCs
                      MFCCs=librosa.feature.mfcc(signal,n_mfcc=40, hop_length=512, n_fft=2048)
                      #extracting data
                      data["labels"].append(i-1)
                      data["MFCCs"].append(MFCCs.T.tolist())
                      data["files"].append(file_path)
                      print(f"{file_path}:{i-1}")
    #store on json file
    with open(json_path,"w") as fp:
        json.dump(data, fp, indent=4)
if __name__=="__main__":
        prepare_dataset(dataset_path, JSON_PATH)
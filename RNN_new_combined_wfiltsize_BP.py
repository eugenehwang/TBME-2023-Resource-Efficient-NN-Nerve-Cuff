# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:09:27 2018

@author: kohr

"""

import Preprocessing_module_BP as PP
import Preprocessing_module_ConPerRing_BP as PP2
import Preprocessing_module_BP_crop_aug as PP_crop
import Preprocessing_module_ConPerRing_BP_crop_aug as PP2_crop
import numpy as np
import scipy
from File_utils import File_utils
import CNN_utils as CNN_utils

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def runRNN_full(NN_hyperparameters,directives):
    foldnum = NN_hyperparameters["foldnum"]
    numfilters = NN_hyperparameters["numfilters"]
    numspikes = NN_hyperparameters["numspikes"]
    Ratnum = directives["ratnum"]
    
    folder = Ratnum + '\\RNN\\CM_CM_C' + str(numfilters)
    folder = folder + '\\'
    
    directives["folder"] = folder
    directives["filename_prefix"] = get_filename_prefix(NN_hyperparameters,
                                                        directives)
    
    folder = folder + '\\'
    print(Ratnum + ': Fold ' + str(foldnum) + ' starting')
    
    #### 1) Loading dataset: Grabbing Spatial/Temporal Emphasis data
    Int_model_sp = None
    RAT_data_sp = None
    
    Int_model_tp = None
    RAT_data_tp = None
    
    RAT_data_sp = PP.Preprocessing_module(Ratnum,foldnum,is_RNN=(True))
    RAT_data_sp.Randomize_Train_and_getValidSet(RAT_data_sp.training_set,RAT_data_sp.training_labels,numspikes)
    RAT_data_sp.Reshape_data_set_RNN()
    RAT_data_tp = PP2.Preprocessing_module(Ratnum,foldnum,is_RNN=(True))
    RAT_data_tp.Randomize_Train_and_getValidSet2(RAT_data_tp.training_set,RAT_data_tp.training_labels,RAT_data_sp.samples1,RAT_data_sp.samples2,
                                         RAT_data_sp.samples3,numspikes)
    RAT_data_tp.Reshape_data_set_RNN()
   
    print(Ratnum + ': Fold ' + str(foldnum) + ' Spatial and Temporal Emphasis data loaded')
    
    ### 2) Call neural network trainer
    train_NN(RAT_data_sp,RAT_data_tp,NN_hyperparameters,directives)
        
    for i in range(20):
        K.clear_session()
        
    print(Ratnum + ': Fold ' + str(foldnum) + ' done')
    
    return None


def get_filename_prefix(NN_hyperparameters,directives,prefix = ""):
    foldnum = NN_hyperparameters.get("foldnum")
    dense_neurons = NN_hyperparameters.get("dense_neurons")
    dropout_rate = NN_hyperparameters.get("dropout_rate")
    filtsize = NN_hyperparameters.get("filtsize")
    channel_width_multiplier = NN_hyperparameters.get("channel_width_multiplier")
    num_1x1s = NN_hyperparameters.get("num_1x1s")
    num_layers = NN_hyperparameters.get("num_layers")
    layer_type_enum = NN_hyperparameters.get("layer_type_enum")
    activation_enum = NN_hyperparameters.get("activation_enum")
    loss_function_enum = NN_hyperparameters.get("loss_function_enum", None)
    use_attention = NN_hyperparameters.get("use_attention", False)
    
    Ratnum = directives["ratnum"]
    
    filename_prefix = (Ratnum + '_DF_PF_Prick_wnoise_CM_CM_CDD_RNN' + prefix 
                       + '_fold' + str(foldnum))

    if num_layers != None: 
        filename_prefix = filename_prefix + 'num_layers' + str(num_layers)
            
    if layer_type_enum != None:
        filename_prefix = filename_prefix + layer_type_enum.name
        
    if activation_enum != None:
        filename_prefix = filename_prefix + 'actv' + activation_enum.name
        
    filename_prefix = filename_prefix + '_denseneur' + str(dense_neurons)

    if loss_function_enum:
        filename_prefix += '_' + loss_function_enum.name
    
    if use_attention:
        filename_prefix += '_attn'
    
    return filename_prefix

def train_NN(RAT_data_sp,RAT_data_tp,NN_hyperparameters,directives):
    load_combined = NN_hyperparameters.get("load_combined")
    filename_prefix = directives.get("filename_prefix")
    folder = directives.get("folder")
    
    # Create RNN
    [labels_training, 
     labels_test, 
     labels_valid] = CNN_utils.get_formatted_labels(
        RAT_data_tp.training_labels,
        RAT_data_tp.test_labels,
        RAT_data_tp.valid_labels,get_labels=True)
         
    num_subclasses = 1
    use_softmax = True
    model = CNN_utils.create_RNN(NN_hyperparameters,RAT_data_tp)
    
        
    directives["inputs_list"] = [CNN_utils.Network.SPATIAL]
    
    [history, class_probs, model] = CNN_utils.train_model(model,NN_hyperparameters,
                                                   directives,RAT_data_sp,RAT_data_tp)
        
    
    if not load_combined:
    
        # Let's save the teacher
        File_utils.save_files(folder,filename_prefix,class_probs,RAT_data_sp.test_labels,history,Full_model=model)
    
    del model
    
    K.clear_session()
    
    return None

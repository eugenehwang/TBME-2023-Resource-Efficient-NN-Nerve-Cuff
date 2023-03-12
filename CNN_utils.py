# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 2021

@author: hwangyic
"""




import tensorflow
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Layer, Lambda, concatenate, Input,Dense, Dropout,\
    Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
        BatchNormalization, LSTM, GRU, Bidirectional
from tensorflow.keras.losses import categorical_crossentropy as cat_crossent
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import File_utils_main
from File_utils_main import Utils
import math

from enum import Enum

class Network(Enum):
    SPATIAL = 1
    TEMPORAL = 2
    COMBINED = 3
    CNN = 4
    RNN = 5

def get_formatted_labels(training_labels,test_labels,valid_labels,get_labels):
    '''
    Returns 
    -------
    One-hot encoded labels for training, test and validation sets

    '''
    labels_training = None
    labels_test = None
    labels_valid = None
    
    if get_labels:
        labels_training = to_categorical(training_labels - 1, None)
        labels_test = to_categorical(test_labels - 1, None)
        labels_valid = to_categorical(valid_labels - 1, None)
    
    return [labels_training, labels_test, labels_valid]




def create_RNN(NN_hyperparameters,RAT_data):
    numfilters = NN_hyperparameters.get("numfilters")
    dense_neurons = NN_hyperparameters.get("dense_neurons")
    filtsize = NN_hyperparameters.get("filtsize")
    use_softmax = NN_hyperparameters.get("use_softmax",True)
    num_layers = NN_hyperparameters.get("num_layers", 3)
    use_attention = NN_hyperparameters.get("use_attention", False)
    if num_layers == None: num_layers = 3
    layer_type_enum = NN_hyperparameters.get("layer_type_enum", Utils.LayerType.LSTM)
    if layer_type_enum == None: layer_type_enum = Utils.LayerType.LSTM
    activation_enum = NN_hyperparameters.get("activation_enum", Utils.Activation.TANH)
    if activation_enum == None: activation_enum = Utils.Activation.TANH
    activation = get_activation(activation_enum)
    data_dimension = RAT_data.numcons
    num_timesteps = 100
    
    model = Sequential()
    model.add(Input(shape=(num_timesteps,data_dimension), name="input"))
    
    def get_RNN_layer(numfilters, return_sequences, input_shape, name, layer_type_enum,
                      activation):
        if layer_type_enum == Utils.LayerType.LSTM:
            return LSTM(numfilters, return_sequences = return_sequences,
                        input_shape = input_shape,
                        name = name, activation = activation)
        elif layer_type_enum == Utils.LayerType.GRU:
            return GRU(numfilters, return_sequences = return_sequences, 
                       input_shape = input_shape,
                       name = name, activation = activation)
        elif layer_type_enum == Utils.LayerType.BIDIR_LSTM:
            return Bidirectional(LSTM(numfilters, return_sequences = return_sequences,
                        input_shape = input_shape,
                        name = name, activation = activation))
        elif layer_type_enum == Utils.LayerType.BIDIR_GRU:
            return Bidirectional(GRU(numfilters, return_sequences = return_sequences,
                        input_shape = input_shape,
                        name = name, activation = activation))
        
    
    model.add(get_RNN_layer(numfilters, return_sequences=True, 
                            name = layer_type_enum.name + "1",
                            input_shape = (num_timesteps, data_dimension),
                            layer_type_enum = layer_type_enum,
                            activation = activation))
    
    for i in range(1, num_layers-1):
        model.add(get_RNN_layer(numfilters, return_sequences=True,
                   input_shape=(num_timesteps, data_dimension), 
                   name = layer_type_enum.name + str(i+1), 
                  layer_type_enum = layer_type_enum,
                  activation = activation))
    
    if use_attention:
        model.add(File_utils_main.attention())
    else:
        model.add(get_RNN_layer(numfilters, return_sequences = False,
                   input_shape = (num_timesteps, data_dimension), 
                   name = layer_type_enum.name + str(num_layers),
                   layer_type_enum = layer_type_enum,
                   activation = activation))
    
    
    
    model.add(Dense(dense_neurons, activation='relu', name = "den1"))
    
    model.add(Dense(3, name = "classification"))
    
    if use_softmax:
        model.add(Activation('softmax', name = "softmax"))
    
    return model

def create_fully_conv_model(numfilters,dense_neurons,dropout_rate,filtsize,
                        num_subclasses=1,use_softmax=True,
                        stack_conv=1,channel_width_multiplier=1,
                        use_1x1=False,num_1x1s=1,fully_conv=False,num_layer=3):
    input_img = Input(shape=(None,None,1), name="input")
    x = input_img
    # Layer 1
    for i in range(1,(stack_conv)+1):
        x = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                   padding='same', name = "conv1-" + str(i))(x)

    
    x = MaxPooling2D((2, 2), padding='same', name = "MP1")(x)

    # Layer 2
    for i in range(1,stack_conv+1):
        x = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                   padding='same', name = "conv2-" + str(i))(x)

    x = Conv2D(numfilters, (1, 1), activation='relu', 
               padding='same', name = "conv1x1_2")(x)

    # Layer 3
    if num_layer > 2:
        x = MaxPooling2D((2, 2), padding='same', name = "MP2")(x)
        
        for i in range(1,math.ceil(stack_conv/2)+1):
            x = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                       padding='same', name = "conv3-" + str(i))(x)
    
        if num_1x1s == 1:
            x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                       name = "conv1x1_3")(x)
        elif num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                       name = "conv1x1_3" + str(i))(x)
    
    # Layer 4
    if num_layer > 3:
        x = MaxPooling2D((2, 2), padding='same', name = "MP3")(x)
        
        for i in range(1,math.ceil(stack_conv/2)+1):
            x = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                       padding='same', name = "conv4-" + str(i))(x)
    
        if num_1x1s == 1:
            x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                       name = "conv1x1_4")(x)
        elif num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                       name = "conv1x1_4" + str(i))(x)
    
    x = Conv2D(3, (1, 1), name = "conv_1x1_class")(x)
    x = GlobalAveragePooling2D()(x)
    if use_softmax:
        x = Activation('softmax')(x)
    
    return Model(inputs=[input_img], outputs = x)

'''

stack_conv: If we want to have multiple convolutional layers before a pooling 
layer is encountered.

'''
def create_single_model(numfilters,dense_neurons,dropout_rate,filtsize,
                        num_subclasses=1,use_softmax=True,
                        stack_conv=1,channel_width_multiplier=1,
                        use_1x1=False,num_1x1s=1,fully_conv=False):
    input_img = Input(shape=(56,100,1), name="input")
    x = input_img
    
    for i in range(1,(stack_conv)+1):
        x = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                   padding='same', name = "conv1-" + str(i))(x)

    x = MaxPooling2D((2, 2), padding='same', name = "MP1")(x)

    for i in range(1,stack_conv+1):
        x = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                   padding='same', name = "conv2-" + str(i))(x)

    if use_1x1:
        x = Conv2D(numfilters, (1, 1), activation='relu', 
                   padding='same', name = "conv1x1_2")(x)
        
    x = MaxPooling2D((2, 2), padding='same', name = "MP2")(x)
    
    for i in range(1,math.ceil(stack_conv/2)+1):
        x = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                   padding='same', name = "conv3-" + str(i))(x)

    if use_1x1 and num_1x1s == 1:
        x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                   name = "conv1x1_3")(x)
    elif use_1x1 and num_1x1s > 1:
        for i in range(1,num_1x1s+1):
            x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                   name = "conv1x1_3" + str(i))(x)
            
    
    if use_1x1:
        pass
        # x = Conv2D(dense_neurons*2, (1, 1), activation='relu', 
        #             padding='same', name = "conv_1x1_end")(x)
    else:
        x = Flatten()(x)
        x = Dense(dense_neurons, activation='relu', name = "Den1")(x)
        x = Dropout(dropout_rate)(x)
    
    if fully_conv:
        x = Conv2D(3, (1, 1), name = "conv_1x1_class")(x)
        x = GlobalAveragePooling2D()(x)
    elif use_1x1:
        x = GlobalAveragePooling2D()(x)
        x = Dense(3)(x)
    else:
        if num_subclasses > 1:
            x = Dense(3*num_subclasses, name = "Den2")(x) 
        else:
            x = Dense(3, name = "Den2")(x) 
    
    if use_softmax:
        x = Activation('softmax')(x)
    
    return Model(inputs=[input_img], outputs = x)

def create_combined_fully_conv_model(numfilters,dense_neurons,dropout_rate,filtsize,
                          num_subclasses=1,stack_conv=1,channel_width_multiplier=1,
                          use_1x1=False, use_softmax=True,num_1x1s=1,fully_conv=False,num_layer=3):
    input1_img = Input(shape=(None,None,1))
    input2_img = Input(shape=(None,None,1))
    
    x1 = input1_img
    x2 = input2_img
    
    for i in range(1,(stack_conv)+1):
        x1 = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv11-" + str(i))(x1)
        x2 = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv21-" + str(i))(x2)

    x1 = MaxPooling2D((2, 2), padding='same', name = "MP11")(x1)
    x2 = MaxPooling2D((2, 2), padding='same', name = "MP21")(x2)
    
    for i in range(1,stack_conv+1):
        x1 = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv12-" + str(i))(x1)
        x2 = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv22-" + str(i))(x2)
    
    x1 = Conv2D(numfilters, (1, 1), activation='relu', 
                padding='same', name = "conv1_1x1_2")(x1)
    x2 = Conv2D(numfilters, (1, 1), activation='relu', 
               padding='same', name = "conv2_1x1_2")(x2)
    
    if num_layer > 2:
        x1 = MaxPooling2D((2, 2), padding='same', name = "MP12")(x1)
        x2 = MaxPooling2D((2, 2), padding='same', name = "MP22")(x2)
        
        for i in range(1,math.ceil(stack_conv/2)+1):
            x1 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                        padding='same', name = "conv13-" + str(i))(x1)
            x2 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                        padding='same', name = "conv23-" + str(i))(x2) 
    
        if num_1x1s == 1:
            x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv1_1x1_3")(x1)
            x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv2_1x1_3")(x2)
        elif num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                            name = "conv1_1x1_3" + str(i))(x1)
                x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                            name = "conv2_1x1_3" + str(i))(x2)
    
    if num_layer > 3:
        x1 = MaxPooling2D((2, 2), padding='same', name = "MP13")(x1)
        x2 = MaxPooling2D((2, 2), padding='same', name = "MP23")(x2)
        
        for i in range(1,math.ceil(stack_conv/2)+1):
            x1 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                        padding='same', name = "conv14-" + str(i))(x1)
            x2 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                        padding='same', name = "conv24-" + str(i))(x2) 
    
        if num_1x1s == 1:
            x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv1_1x1_4")(x1)
            x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv2_1x1_4")(x2)
        elif num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                            name = "conv1_1x1_4" + str(i))(x1)
                x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                            name = "conv2_1x1_4" + str(i))(x2)
    
    x = tensorflow.keras.layers.concatenate([x1, x2])   


    x = Conv2D(dense_neurons, (1, 1), activation='relu', 
               padding='same', name = "conv_1x1_end")(x)
    
    x = Conv2D(3, (1, 1), name = "conv_1x1_class")(x)
    x = GlobalAveragePooling2D()(x)
    
    if use_softmax:
        x = Activation('softmax')(x)
    
    return Model(inputs=[input1_img, input2_img], outputs = x)

def create_combined_model(numfilters,dense_neurons,dropout_rate,filtsize,
                          num_subclasses=1,stack_conv=1,channel_width_multiplier=1,
                          use_1x1=False, use_softmax=True,num_1x1s=1,fully_conv=False):
    input1_img = Input(shape=(56,100,1))
    input2_img = Input(shape=(56,100,1))
    
    x1 = input1_img
    x2 = input2_img
    
    # First Convolutional layer(s)
    for i in range(1,(stack_conv)+1):
        x1 = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv11-" + str(i))(x1)
        x2 = Conv2D(numfilters, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv21-" + str(i))(x2)

    x1 = MaxPooling2D((2, 2), padding='same', name = "MP11")(x1)
    x2 = MaxPooling2D((2, 2), padding='same', name = "MP21")(x2)
    
    # Second Convolutional layer(s)
    for i in range(1,stack_conv+1):
        x1 = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv12-" + str(i))(x1)
        x2 = Conv2D(numfilters*channel_width_multiplier, (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv22-" + str(i))(x2)
    
    # If using fully convolutional architecture, want to add an extra 1x1 layer 
    # to allow for more complex functions to be represented
    if use_1x1:
        x1 = Conv2D(numfilters, (1, 1), activation='relu', 
                    padding='same', name = "conv1_1x1_2")(x1)
        x2 = Conv2D(numfilters, (1, 1), activation='relu', 
                   padding='same', name = "conv2_1x1_2")(x2)
    
    x1 = MaxPooling2D((2, 2), padding='same', name = "MP12")(x1)
    x2 = MaxPooling2D((2, 2), padding='same', name = "MP22")(x2)
    
    # Third Convolutional layer(s)
    for i in range(1,math.ceil(stack_conv/2)+1):
        x1 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv13-" + str(i))(x1)
        x2 = Conv2D(numfilters*(channel_width_multiplier**2), (filtsize, filtsize), activation='relu', 
                    padding='same', name = "conv23-" + str(i))(x2) 
    
    
    # If using fully convolutional architecture, want to add an extra 1x1 layer 
    # to allow for more complex functions to be represented
    if use_1x1 and num_1x1s == 1:
        x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                    name = "conv1_1x1_3")(x1)
        x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                    name = "conv2_1x1_3")(x2)
    elif use_1x1 and num_1x1s > 1:
        for i in range(1,num_1x1s+1):
            x1 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv1_1x1_3" + str(i))(x1)
            x2 = Conv2D(dense_neurons, (1, 1), activation='relu', 
                        name = "conv2_1x1_3" + str(i))(x2)
    
    x = tensorflow.keras.layers.concatenate([x1, x2])   
    
    if use_1x1:
        x = Conv2D(dense_neurons, (1, 1), activation='relu', 
                    padding='same', name = "conv_1x1_end")(x)
    else:
        x = Flatten()(x)
        x = Dense(dense_neurons*2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    if fully_conv:
        x = Conv2D(3, (1, 1), name = "conv_1x1_class")(x)
        x = GlobalAveragePooling2D()(x)
    elif use_1x1:
        x = GlobalAveragePooling2D()(x)
        x = Dense(3)(x)
    else:
        if num_subclasses > 1:
            x = Dense(3*num_subclasses)(x) 
        else:
            x = Dense(3)(x) 
    
    if use_softmax:
        x = Activation('softmax')(x)
    
    return Model(inputs=[input1_img, input2_img], outputs = x)

def transfer_weights(Int_model1, Int_model2, Combined_model,stack_conv = 1,
                     use_1x1=False,num_1x1s=1,fully_conv=False,num_layer=3):
    '''
    All 3 models must have the same number of neurons for 
    convolutional/dense layers.
    
    Transfer pre-trained weights from Int_model1 and Int_model2 
    to Combined_model
    
    Parameters
    ----------
    Int_model1 : Keras Model object
        Spatial Emphasis model. 
    Int_model2 : Keras Model object
        Temporal Emphasis model.
    Combined_model : Keras Model object
        Model combining spatial/temporal emphasis models.

    Returns
    -------
    None.

    '''
    
    for i in range(1,(stack_conv)+1):
        C11_w = Int_model1.get_layer("conv1-" + str(i)).get_weights()
        C21_w = Int_model2.get_layer("conv1-" + str(i)).get_weights()
        Combined_model.get_layer("conv11-" + str(i)).set_weights(C11_w)
        Combined_model.get_layer("conv21-" + str(i)).set_weights(C21_w)
        
    for i in range(1,stack_conv+1):
        C12_w = Int_model1.get_layer("conv2-" + str(i)).get_weights()
        C22_w = Int_model2.get_layer("conv2-" + str(i)).get_weights()
        Combined_model.get_layer("conv12-" + str(i)).set_weights(C12_w)
        Combined_model.get_layer("conv22-" + str(i)).set_weights(C22_w)
    
    if num_layer > 2:
        for i in range(1,math.ceil(stack_conv/2)+1):
            C13_w = Int_model1.get_layer("conv3-" + str(i)).get_weights()
            C23_w = Int_model2.get_layer("conv3-" + str(i)).get_weights()   
            Combined_model.get_layer("conv13-" + str(i)).set_weights(C13_w)
            Combined_model.get_layer("conv23-" + str(i)).set_weights(C23_w)     
            
    if num_layer > 3:
        for i in range(1,math.ceil(stack_conv/2)+1):
            C14_w = Int_model1.get_layer("conv4-" + str(i)).get_weights()
            C24_w = Int_model2.get_layer("conv4-" + str(i)).get_weights()   
            Combined_model.get_layer("conv14-" + str(i)).set_weights(C14_w)
            Combined_model.get_layer("conv24-" + str(i)).set_weights(C24_w)     

    if use_1x1:
        C1_1x1_2_w = Int_model1.get_layer("conv1x1_2").get_weights()
        C2_1x1_2_w = Int_model2.get_layer("conv1x1_2").get_weights()   
        Combined_model.get_layer("conv1_1x1_2").set_weights(C1_1x1_2_w)
        Combined_model.get_layer("conv2_1x1_2").set_weights(C2_1x1_2_w)
        
    if num_layer > 2:
        if use_1x1 and num_1x1s == 1:
            C1_1x1_3_w = Int_model1.get_layer("conv1x1_3").get_weights()
            C2_1x1_3_w = Int_model2.get_layer("conv1x1_3").get_weights()   
            Combined_model.get_layer("conv1_1x1_3").set_weights(C1_1x1_3_w)
            Combined_model.get_layer("conv2_1x1_3").set_weights(C2_1x1_3_w)
        elif use_1x1 and num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                C1_1x1_3_w = Int_model1.get_layer("conv1x1_3" + str(i)).get_weights()
                C2_1x1_3_w = Int_model2.get_layer("conv1x1_3" + str(i)).get_weights()   
                Combined_model.get_layer("conv1_1x1_3" + str(i)).set_weights(C1_1x1_3_w)
                Combined_model.get_layer("conv2_1x1_3" + str(i)).set_weights(C2_1x1_3_w)
                
    if num_layer > 3:
        if use_1x1 and num_1x1s == 1:
            C1_1x1_4_w = Int_model1.get_layer("conv1x1_4").get_weights()
            C2_1x1_4_w = Int_model2.get_layer("conv1x1_4").get_weights()   
            Combined_model.get_layer("conv1_1x1_4").set_weights(C1_1x1_4_w)
            Combined_model.get_layer("conv2_1x1_4").set_weights(C2_1x1_4_w)
        elif use_1x1 and num_1x1s > 1:
            for i in range(1,num_1x1s+1):
                C1_1x1_4_w = Int_model1.get_layer("conv1x1_4" + str(i)).get_weights()
                C2_1x1_4_w = Int_model2.get_layer("conv1x1_4" + str(i)).get_weights()   
                Combined_model.get_layer("conv1_1x1_4" + str(i)).set_weights(C1_1x1_4_w)
                Combined_model.get_layer("conv2_1x1_4" + str(i)).set_weights(C2_1x1_4_w)
        
    M11_w = Int_model1.get_layer("MP1").get_weights()
    M21_w = Int_model2.get_layer("MP1").get_weights()
    Combined_model.get_layer("MP11").set_weights(M11_w)
    Combined_model.get_layer("MP21").set_weights(M21_w)
    
    if num_layer > 2:
        
        M12_w = Int_model1.get_layer("MP2").get_weights()
        M22_w = Int_model2.get_layer("MP2").get_weights()
        Combined_model.get_layer("MP12").set_weights(M12_w)
        Combined_model.get_layer("MP22").set_weights(M22_w)
        
    if num_layer > 3:
        
        M13_w = Int_model1.get_layer("MP3").get_weights()
        M23_w = Int_model2.get_layer("MP3").get_weights()
        Combined_model.get_layer("MP13").set_weights(M13_w)
        Combined_model.get_layer("MP23").set_weights(M23_w)
    
    
    
    


def fit_cropped_dataset(model,training_set,training_labels,valid_set,valid_labels,
                        epochs,batch_size,shuffle,early_stopping_monitor=None):
    
    num_crops = 2
    for i in range(num_crops):
        training_set_subset = None
        training_labels_subset = None
        valid_set_subset = None
        valid_labels_subset = None
        if isinstance(training_set,(list)):
            training_set_size = training_set[0].shape[0]
            valid_set_size = valid_set[0].shape[0]
            
            train_start_ind = int(i*(training_set_size/num_crops))
            train_stop_ind = int((i+1)*(training_set_size/num_crops))
            valid_start_ind = int(i*(valid_set_size/num_crops))
            valid_stop_ind = int((i+1)*(valid_set_size/num_crops))
            
            training_set_subset = [training_set[0][train_start_ind:train_stop_ind,:,:,:],
                                   training_set[1][train_start_ind:train_stop_ind,:,:,:]]
            training_labels_subset = training_labels[train_start_ind:train_stop_ind,:]
            valid_set_subset = [valid_set[0][valid_start_ind:valid_stop_ind,:,:,:],
                                valid_set[1][valid_start_ind:valid_stop_ind,:,:,:]]
            valid_labels_subset = valid_labels[valid_start_ind:valid_stop_ind,:]
        else:
            training_set_size = training_set.shape[0]
            valid_set_size = valid_set.shape[0]
            
            train_start_ind = int(i*(training_set_size/num_crops))
            train_stop_ind = int((i+1)*(training_set_size/num_crops))
            valid_start_ind = int(i*(valid_set_size/num_crops))
            valid_stop_ind = int((i+1)*(valid_set_size/num_crops))
            
            training_set_subset = training_set[train_start_ind:train_stop_ind,:,:,:]
            training_labels_subset = training_labels[train_start_ind:train_stop_ind,:]
            valid_set_subset = valid_set[valid_start_ind:valid_stop_ind,:,:,:]
            valid_labels_subset = valid_labels[valid_start_ind:valid_stop_ind,:]
            
        
        if early_stopping_monitor == None:
            model.fit(training_set_subset, training_labels_subset,
                      epochs=epochs,batch_size=batch_size,shuffle=shuffle,
                      validation_data=(valid_set_subset, 
                                       valid_labels_subset))
        else:
            model.fit(training_set_subset, training_labels_subset,
                      epochs=epochs,batch_size=batch_size,shuffle=shuffle,
                      validation_data=(valid_set_subset, 
                                       valid_labels_subset),
                      callbacks=[early_stopping_monitor])

def train_model(model,NN_hyperparameters,directives,RAT_data_sp,RAT_data_tp,
                numepochs=1000,opt=None):
    batch_size = NN_hyperparameters.get("batch_size")
    valid_patience = NN_hyperparameters.get("valid_patience")
    min_delta = NN_hyperparameters.get("min_delta")
    restore_best_weights = NN_hyperparameters.get("restore_best_weights")
    loss_function_enum = NN_hyperparameters.get("loss_function_enum",None)
    train_with_crops = directives.get("train_with_crops",False)
    load_combined = directives.get("load_combined",False)
    inputs_list = directives.get("inputs_list")
    print_model = directives.get("print_model",True)
    
    if print_model:
        print(model.summary())
    
    early_stopping_monitor = None
    callbacks = None
    if valid_patience > 0:
        early_stopping_monitor = EarlyStopping(patience=valid_patience,
                                               min_delta=min_delta,
                                               restore_best_weights=(restore_best_weights))
        callbacks = [early_stopping_monitor]
        
    # Build up datasets
    training_set = []
    valid_set = []
    test_set = []
    for input_type in inputs_list:
        if input_type == Network.SPATIAL:
            if train_with_crops:
                training_set.append(RAT_data_sp.training_set_cropped)
                valid_set.append(RAT_data_sp.valid_set_cropped)
            else:
                training_set.append(RAT_data_sp.training_set)
                valid_set.append(RAT_data_sp.valid_set)
            test_set.append(RAT_data_sp.test_set)
        elif input_type == Network.TEMPORAL:
            if train_with_crops:
                training_set.append(RAT_data_tp.training_set_cropped)
                valid_set.append(RAT_data_tp.valid_set_cropped)
            else:
                training_set.append(RAT_data_tp.training_set)
                valid_set.append(RAT_data_tp.valid_set)
            test_set.append(RAT_data_tp.test_set)
    
    labels_training = None
    labels_test = None
    labels_valid = None
    
    if train_with_crops:
        [labels_training, 
         labels_test, 
         labels_valid] = get_formatted_labels(
            RAT_data_sp.training_labels_cropped,
            RAT_data_sp.test_labels_cropped,
            RAT_data_sp.valid_labels_cropped,get_labels=True)
    else:
        [labels_training, 
         labels_test, 
         labels_valid] = get_formatted_labels(
            RAT_data_sp.training_labels,
            RAT_data_sp.test_labels,
            RAT_data_sp.valid_labels,get_labels=True)
    
             
    if opt == None:
        opt = SGD(lr=0.005, decay=1e-5, momentum=0.9, nesterov=True)
             
    model.compile(loss=get_loss_function(loss_function_enum),
                  optimizer=opt,
                  metrics=['accuracy',Utils.f1_tf,Utils.macro_f1],)
        
    history = None
    # If we're not loading the model from a file, train it
    if not load_combined:
        if train_with_crops:
            fit_cropped_dataset(model,
                                training_set,labels_training,
                                valid_set,labels_valid,
                                epochs=numepochs,batch_size=batch_size,shuffle=True,
                                early_stopping_monitor=early_stopping_monitor)
        else:
            history = model.fit(training_set, labels_training,
                  epochs=numepochs,batch_size = batch_size, shuffle=True,
                  validation_data=(valid_set, labels_valid), callbacks=callbacks)
            
        score = model.evaluate(test_set, labels_test)
        np.disp(score)
        
    class_probs = model.predict(test_set)
        
    return [history, class_probs, model]

def get_loss_function(loss_function_enum):
    
    
    if loss_function_enum == Utils.LossFunction.CATEGORICAL_CROSSENTROPY or \
        loss_function_enum == None:
        return 'categorical_crossentropy'
    elif loss_function_enum == Utils.LossFunction.MACRO_SOFT_F1:
        return Utils.macro_soft_f1
    elif loss_function_enum == Utils.LossFunction.CCE_MSF1:
        return Utils.cce_msf1
    
    
def get_activation(activation_enum):
    
    
    if activation_enum == Utils.Activation.ReLU or \
        activation_enum == None:
        return 'relu'
    elif activation_enum == Utils.Activation.TANH:
        return 'tanh'
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:09:27 2018

@author: kohr

"""

import Preprocessing_module_BP as PP
import Preprocessing_module_ConPerRing_BP as PP2
import Preprocessing_module_BP_crop_aug as PP_crop
import Preprocessing_module_ConPerRing_BP_crop_aug as PP2_crop
from File_utils import File_utils
import CNN_utils as CNN_utils
import numpy as np
import scipy
import sys


import tensorflow as tf

from tensorflow.math import log, exp
from tensorflow import reduce_sum, reduce_mean

from tensorflow import matmul

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, concatenate, Input,Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy as cat_crossent
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from enum import Enum

class Network(Enum):
    SPATIAL = 1
    TEMPORAL = 2
    COMBINED = 3

def runCNN_full(Ratnum,foldnum,numepochs,size_batch,valid_patience,min_delta,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,print_model,load_se_te,load_combined,load_student,stack_conv,channel_width_multiplier,train_with_crops,num_layer):
    
    num_1x1s=1
    fully_conv=True
    
    
    folder = Ratnum + '\\CM_CM_C' + str(numfilters) + '_stack' 
    if num_layer == 2:
        folder += str(stack_conv) + '-' + str(stack_conv)
    elif num_layer == 3:
        folder += str(stack_conv) + '-' + str(stack_conv) + '-' + str(int(stack_conv/2))
    elif num_layer == 4:
        folder += str(stack_conv) + '-' + str(stack_conv) + '-' + str(int(stack_conv/2)) + '-' + str(int(stack_conv/2))
    
    if train_with_crops:
        folder = folder + "_crop"
    folder = folder + '\\'
    print(Ratnum + ': Fold ' + str(foldnum) + ' starting')
    
    #### Grabbing Spatial/Temporal Emphasis data
    
    Int_model_sp = None
    RAT_data_sp = None
    
    Int_model_tp = None
    RAT_data_tp = None
    if train_with_crops:
        RAT_data_sp = PP_crop.Preprocessing_module(Ratnum,foldnum)
        RAT_data_sp.Randomize_Train_and_getValidSet(RAT_data_sp.training_set,RAT_data_sp.training_labels,numspikes)
        # Get cropped dataset, with 4x the amount of data, so 4x the amount of validation points needed 
        RAT_data_sp.Randomize_Train_and_getValidSet(RAT_data_sp.training_set_cropped,RAT_data_sp.training_labels_cropped,numspikes*4,update_cropped=True)
        RAT_data_tp = PP2_crop.Preprocessing_module(Ratnum,foldnum)
        # RAT_data_tp.Reshape_data_set(RAT_data_tp.training_set,RAT_data_tp.test_set)
        RAT_data_tp.Randomize_Train_and_getValidSet2(RAT_data_tp.training_set,RAT_data_tp.training_labels,RAT_data_sp.samples1,RAT_data_sp.samples2,
                                             RAT_data_sp.samples3,numspikes)
        # Get cropped dataset, with 4x the amount of data, so 4x the amount of validation points needed 
        RAT_data_tp.Randomize_Train_and_getValidSet2(RAT_data_tp.training_set_cropped,RAT_data_tp.training_labels_cropped,RAT_data_sp.samples1_cropped,RAT_data_sp.samples2_cropped,
                                             RAT_data_sp.samples3_cropped,numspikes*4,update_cropped=True)
    else:
        RAT_data_sp = PP.Preprocessing_module(Ratnum,foldnum)
        RAT_data_sp.Randomize_Train_and_getValidSet(RAT_data_sp.training_set,RAT_data_sp.training_labels,numspikes)
        RAT_data_tp = PP2.Preprocessing_module(Ratnum,foldnum)
        RAT_data_tp.Reshape_data_set(RAT_data_tp.training_set,RAT_data_tp.test_set)
        RAT_data_tp.Randomize_Train_and_getValidSet2(RAT_data_tp.training_set,RAT_data_tp.training_labels,RAT_data_sp.samples1,RAT_data_sp.samples2,
                                             RAT_data_sp.samples3,numspikes)
       
    print(Ratnum + ': Fold ' + str(foldnum) + ' Spatial and Temporal Emphasis data loaded')
    
    ### Train Spatial/Temporal Emphasis models
    prefix_se = ''
    prefix_te = '_ConPerRing'
    if not load_combined:
        Int_model_sp = CNN_CM_CM_DD(Ratnum,foldnum,folder,RAT_data_sp,RAT_data_tp,
                                    Network.SPATIAL,prefix_se,numepochs,size_batch,
                                    valid_patience,min_delta,numspikes,numfilters,
                                    filtsize,dropout_rate,dense_neurons,print_model,
                                    load_se_te,stack_conv,channel_width_multiplier,
                                    num_1x1s=num_1x1s,fully_conv=fully_conv,
                                    train_with_crops=train_with_crops,num_layer=num_layer)
        print(Ratnum + ': Fold ' + str(foldnum) + ' spatial_done')
        Int_model_tp = CNN_CM_CM_DD(Ratnum,foldnum,folder,RAT_data_sp,RAT_data_tp,
                                    Network.TEMPORAL,prefix_te,numepochs,size_batch,
                                    valid_patience,min_delta,numspikes,numfilters,
                                    filtsize,dropout_rate,dense_neurons,print_model,
                                    load_se_te,stack_conv,channel_width_multiplier,
                                    num_1x1s=num_1x1s,fully_conv=fully_conv,
                                    train_with_crops=train_with_crops,num_layer=num_layer)
        print(Ratnum + ': Fold ' + str(foldnum) + ' temporal_done')

    #### Beginning Combined Model
    
    combined_models_to_dense(folder,Int_model_sp,Int_model_tp,RAT_data_sp,RAT_data_tp,
                             Ratnum,foldnum,numepochs,size_batch,valid_patience,
                             min_delta,numspikes,numfilters,filtsize, dropout_rate,
                             dense_neurons,print_model,load_combined,load_student,
                             stack_conv,channel_width_multiplier,
                             num_1x1s=num_1x1s,fully_conv=fully_conv,
                             train_with_crops=train_with_crops,num_layer=num_layer)
    
    for i in range(15):
        K.clear_session()
        
    print(Ratnum + ': Fold ' + str(foldnum) + ' done')
    
    return None

def get_filename_prefix(Ratnum,prefix,foldnum,filtsize,dropout_rate,dense_neurons,channel_width_multiplier, num_1x1s, fully_conv):
    return Ratnum + '_DF_PF_Prick_wnoise_CM_CM_CDD' + prefix + '_fold' + str(foldnum) + '_filtsize_' + str(filtsize) + '_dropoutrate_' + str(dropout_rate) + '_denseneur' + str(dense_neurons) + '_cwm' + str(channel_width_multiplier) + '_1x1s' + str(num_1x1s)

def runCNN_X_times(Ratnum,foldnum,numepochs,size_batch,valid_patience,min_delta,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,student_numfilters,student_densefilters,print_model,load_se_te,load_combined,load_student,temperature,temperature_teacher,lambda_teacher,alpha_distill,lambda_distill,num_subclasses):
    
    return

def CNN_CM_CM_DD(Ratnum,foldnum,folder,RAT_data,RAT_data2,network_type,prefix,numepochs,size_batch,valid_patience,min_delta,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,print_model,load_se_te,
                 stack_conv,channel_width_multiplier,num_1x1s=1,fully_conv=False,
                 train_with_crops=False,num_layer=3):

    ''' Metric functions'''
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall))
    
    """
    CNN code
    """
    filename_prefix = get_filename_prefix(Ratnum,prefix,foldnum,filtsize,dropout_rate,dense_neurons,channel_width_multiplier,num_1x1s,fully_conv)
    
    dependencies = {
         'f1': f1
    }
    
    
    if load_se_te:
        return File_utils.load_model(folder,filename_prefix,dependencies,int_model=True)
    
    """
    Labels and Training Reformating
    """
    
    labels_training = None
    labels_test = None
    labels_valid = None
    
    labels_training = to_categorical(RAT_data.training_labels - 1, None)
    labels_test = to_categorical(RAT_data.test_labels - 1, None)
    labels_valid = to_categorical(RAT_data.valid_labels - 1, None)
        
    labels_training_cropped = None
    labels_test_cropped = None
    labels_valid_cropped = None
    
    if train_with_crops:
        labels_training_cropped = to_categorical(RAT_data.training_labels_cropped - 1, None)
        labels_test_cropped = to_categorical(RAT_data.test_labels_cropped - 1, None)
        labels_valid_cropped = to_categorical(RAT_data.valid_labels_cropped - 1, None)
    
    num_subclasses = 1
    use_softmax = True
    Full_model = CNN_utils.create_fully_conv_model(numfilters, dense_neurons, dropout_rate, filtsize, num_subclasses, use_softmax,
                                               stack_conv=stack_conv,
                                               channel_width_multiplier=channel_width_multiplier,
                                               use_1x1=True,num_1x1s=num_1x1s,fully_conv=fully_conv,num_layer=num_layer)
    
    Intermediate_model = pretrain_single_model(folder,filename_prefix,Full_model,False,numepochs,dense_neurons,
                          num_subclasses,RAT_data,RAT_data2,
                          labels_training,labels_valid,labels_test,
                          labels_training_cropped,labels_valid_cropped,labels_test_cropped,
                          print_model,size_batch,
                          valid_patience,min_delta,f1,network_type,
                          stack_conv=stack_conv,num_1x1s=num_1x1s,fully_conv=fully_conv,
                          train_with_crops=train_with_crops,num_layer=num_layer)
   
    
    class_probs = None
    if network_type == Network.SPATIAL:
        score = Full_model.evaluate(RAT_data.test_set, labels_test)
        np.disp(score)
        class_probs = Full_model.predict(RAT_data.test_set)
    elif network_type == Network.TEMPORAL:
        score = Full_model.evaluate(RAT_data2.test_set, labels_test)
        np.disp(score)
        class_probs = Full_model.predict(RAT_data2.test_set)

    
    File_utils.save_files(folder,filename_prefix,class_probs,RAT_data.test_labels)
    
    
    del Full_model
    
    return Intermediate_model

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

def pretrain_single_model(folder,filename_prefix,Full_model,load_student,
                          numepochs,dense_neurons,num_subclasses,
                          RAT_data,RAT_data2,labels_training,labels_valid,labels_test,
                          labels_training_cropped,labels_valid_cropped,labels_test_cropped,
                          print_model,size_batch,valid_patience,min_delta,f1,network_type,
                          use_distillation = False,teacher=None,
                          temperature=5,alpha=0.9,lambda_distill=0.1,
                          stack_conv=1, num_1x1s=1,fully_conv=False,
                          train_with_crops=False,num_layer=3):
    
    dependencies = {
         'f1': f1
    }
    
    if load_student:
        Intermediate_model = File_utils.load_model(folder,filename_prefix,dependencies,int_model=True)
        
        if Intermediate_model != None:
            return Intermediate_model
    
    input_img = Full_model.get_layer("input").input
    Conv1_out = Full_model.get_layer("conv1-" +str(stack_conv))
    MPool1 = MaxPooling2D((4, 4), padding='same', name = "mpool1")(Conv1_out.output)
    
    x = GlobalAveragePooling2D()(MPool1)
    
    x = Dense(3)(x)
    x = Activation('softmax')(x)
    
    Initial_guess1_model = Model(input_img,x)
    
    
    Initial_guess1 = Initial_guess1_model
    sgd = SGD(lr=0.2, decay=1e-4, momentum=0.9, nesterov=True)
    Initial_guess1.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)

    if print_model:
        print(Initial_guess1.summary())
    
    score = None
    if network_type == Network.SPATIAL:
        if train_with_crops:
            fit_cropped_dataset(Initial_guess1,
                                RAT_data.training_set_cropped,
                                labels_training_cropped,
                                RAT_data.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=10,batch_size=size_batch,shuffle=True)
        else:
            Initial_guess1.fit(RAT_data.training_set, labels_training,
                  epochs=25,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data.valid_set, labels_valid))
        score = Initial_guess1.evaluate(RAT_data.test_set, labels_test)
    elif network_type == Network.TEMPORAL:
        if train_with_crops:
            fit_cropped_dataset(Initial_guess1,
                                RAT_data2.training_set_cropped,
                                labels_training_cropped,
                                RAT_data2.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=25,batch_size=size_batch,shuffle=True)
        else:
            Initial_guess1.fit(RAT_data2.training_set, labels_training,
                  epochs=25,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data2.valid_set, labels_valid))
        score = Initial_guess1.evaluate(RAT_data2.test_set, labels_test)
    np.disp(score)
    
    Conv2_out = Full_model.get_layer("conv1x1_2")
    MPool2 = MaxPooling2D((2, 2), padding='same', name = "mpool2")(Conv2_out.output)
    
    x = GlobalAveragePooling2D()(MPool2)
    x = Dense(3)(x)
    x = Activation('softmax')(x)
    
    
    Initial_guess2_model = Model(input_img,x)
    

    Initial_guess2 = Initial_guess2_model
    sgd = SGD(lr=0.003, decay=1e-5, momentum=0.9, nesterov=True)
    Initial_guess2.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)

    if print_model:
        print(Initial_guess2.summary())
    
    score = None
    if network_type == Network.SPATIAL:
        Initial_guess2.fit(RAT_data.training_set, labels_training,
                  epochs=25,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data.valid_set, labels_valid))
        if train_with_crops:
            fit_cropped_dataset(Initial_guess2,
                                RAT_data.training_set_cropped,
                                labels_training_cropped,
                                RAT_data.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=25,batch_size=size_batch,shuffle=True)
        else:
            Initial_guess2.fit(RAT_data.training_set, labels_training,
                  epochs=25,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data.valid_set, labels_valid))
        score = Initial_guess2.evaluate(RAT_data.test_set, labels_test)
    elif network_type == Network.TEMPORAL:
        if train_with_crops:
            fit_cropped_dataset(Initial_guess2,
                                RAT_data2.training_set_cropped,
                                labels_training_cropped,
                                RAT_data2.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=25,batch_size=size_batch,shuffle=True)
        else:
            Initial_guess2.fit(RAT_data2.training_set, labels_training,
                  epochs=25,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data2.valid_set, labels_valid))
            
        score = Initial_guess2.evaluate(RAT_data2.test_set, labels_test)
    
    np.disp(score)
        
    last_layer_name = "conv1x1_" + str(num_layer)
    
    if num_1x1s > 1:
        last_layer_name = last_layer_name + str(num_1x1s)
    
    
    Intermediate_model = Model(Full_model.get_layer("input").input,Full_model.get_layer(last_layer_name).output)
    
    early_stopping_monitor = None
    if valid_patience > 0:
        early_stopping_monitor = EarlyStopping(patience=valid_patience,
                                               min_delta=min_delta)
    history = None

    
    if print_model:
        print(Full_model.summary())
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Full_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    if network_type == Network.SPATIAL:
        if train_with_crops:
            fit_cropped_dataset(Full_model,
                                RAT_data.training_set_cropped,
                                labels_training_cropped,
                                RAT_data.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=numepochs,batch_size=size_batch,shuffle=True,
                                early_stopping_monitor=early_stopping_monitor)
        else:
            history = Full_model.fit(RAT_data.training_set, labels_training,
                  epochs=numepochs,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data.valid_set, labels_valid), callbacks=[early_stopping_monitor])
        
        score = Full_model.evaluate(RAT_data.test_set, labels_test)
    
    elif network_type == Network.TEMPORAL:
        if train_with_crops:
            fit_cropped_dataset(Full_model,
                                RAT_data2.training_set_cropped,
                                labels_training_cropped,
                                RAT_data2.valid_set_cropped,
                                labels_valid_cropped,
                                epochs=numepochs,batch_size=size_batch,shuffle=True,
                                early_stopping_monitor=early_stopping_monitor)
        else:
            history = Full_model.fit(RAT_data2.training_set, labels_training,
                  epochs=numepochs,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data2.valid_set, labels_valid), callbacks=[early_stopping_monitor])
                
        score = Full_model.evaluate(RAT_data2.test_set, labels_test)
        
    np.disp(score)
        
    File_utils.save_files(folder,filename_prefix,history = history,Intermediate_model=Intermediate_model)
    return Intermediate_model

def combined_models_to_dense(folder,Int_model1,Int_model2,RAT_data,RAT_data2,Ratnum,foldnum,numepochs,size_batch,valid_patience,min_delta,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,print_model,load_combined,load_student,stack_conv,channel_width_multiplier,num_1x1s=1,fully_conv=False,
                             train_with_crops=False,num_layer=3):
    
    ''' Metric functions'''
    def f1_score():
        def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.
        
                Only computes a batch-wise average of recall.
        
                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall
        
            def precision(y_true, y_pred):
                """Precision metric.
        
                Only computes a batch-wise average of precision.
        
                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
            
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall))
        return f1
    
    filename_prefix = get_filename_prefix(Ratnum,'_Combined',foldnum,filtsize,dropout_rate,dense_neurons,channel_width_multiplier,num_1x1s,fully_conv)
    
    
    dependencies = {
         'f1': f1_score()
    }
    
    Combined_model = None
    if load_combined:
        Combined_model = File_utils.load_model(folder,filename_prefix,dependencies)
    else:
        Combined_model = CNN_utils.create_combined_fully_conv_model(numfilters,dense_neurons,dropout_rate,filtsize,
                                                         stack_conv=stack_conv,
                                                         channel_width_multiplier=channel_width_multiplier,
                                                         use_1x1=True,num_1x1s=num_1x1s,fully_conv=fully_conv,num_layer=num_layer)
        
        CNN_utils.transfer_weights(Int_model1, Int_model2, Combined_model,stack_conv,num_1x1s=num_1x1s,fully_conv=fully_conv,num_layer=num_layer)
        
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        
        Combined_model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy',f1_score()],)
        
    
    if print_model:
        print(Combined_model.summary())
    
    labels_training = None
    labels_test = None
    labels_valid = None
    
    labels_training = to_categorical(RAT_data.training_labels - 1, None)
    labels_test = to_categorical(RAT_data.test_labels - 1, None)
    labels_valid = to_categorical(RAT_data.valid_labels - 1, None)
    
    labels_training_cropped = None
    labels_test_cropped = None
    labels_valid_cropped = None
    
    if train_with_crops:
        labels_training_cropped = to_categorical(RAT_data.training_labels_cropped - 1, None)
        labels_test_cropped = to_categorical(RAT_data.test_labels_cropped - 1, None)
        labels_valid_cropped = to_categorical(RAT_data.valid_labels_cropped - 1, None)
    
    
    early_stopping_monitor = None
    if valid_patience > 0:
        early_stopping_monitor = EarlyStopping(patience=valid_patience,min_delta=min_delta)
    
    history = None
    
    if not load_combined:
        if valid_patience > 0:
            if train_with_crops:
                fit_cropped_dataset(Combined_model,
                                    [RAT_data.training_set_cropped, RAT_data2.training_set_cropped],
                                    labels_training_cropped,
                                    [RAT_data.valid_set_cropped, RAT_data2.valid_set_cropped],
                                    labels_valid_cropped,
                                    epochs=numepochs,batch_size=size_batch,shuffle=True,
                                    early_stopping_monitor=early_stopping_monitor)
            else:
                history = Combined_model.fit([RAT_data.training_set, RAT_data2.training_set], labels_training,
                      epochs=numepochs,batch_size = size_batch, shuffle=True,
                      validation_data=([RAT_data.valid_set, RAT_data2.valid_set], labels_valid), callbacks=[early_stopping_monitor])
        else:
            
            if train_with_crops:
                fit_cropped_dataset(Combined_model,
                                    [RAT_data.training_set_cropped, RAT_data2.training_set_cropped],
                                    labels_training_cropped,
                                    [RAT_data.valid_set_cropped, RAT_data2.valid_set_cropped],
                                    labels_valid_cropped,
                                    epochs=numepochs,batch_size=size_batch,shuffle=True)
            else:
                history = Combined_model.fit([RAT_data.training_set, RAT_data2.training_set], labels_training,
                      epochs=numepochs,batch_size = size_batch, shuffle=True,
                      validation_data=([RAT_data.valid_set, RAT_data2.valid_set], labels_valid))
            
        score = Combined_model.evaluate([RAT_data.test_set, RAT_data2.test_set], labels_test)
        np.disp(score)
        
        
    if not load_combined:
        class_probs = Combined_model.predict([RAT_data.test_set, RAT_data2.test_set])
    
        File_utils.save_files(folder,filename_prefix,class_probs,RAT_data.test_labels,history,Full_model=Combined_model)

    
    del Combined_model
    
    K.clear_session()
    
    return None
    
    
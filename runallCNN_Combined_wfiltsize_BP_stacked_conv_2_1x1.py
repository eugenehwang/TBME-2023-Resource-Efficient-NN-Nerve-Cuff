# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:00:22 2018

@author: kohr
"""

import CNN_new_combined_wfiltsize_BP_stacked_conv_2_1x1 as CNN

valid_patience = 15
min_delta = 0
epochs = 1000
batch_size = 100
numspikes = 1000
numfilters_arr = [16]
dense_neurons_arr = [8]
filtsizes = [3]
dropout_rate = 0.5
num_layers = [2]
print_model = True
load_se_te = False
load_combined = False
load_student = False
stack_conv = 2
channel_width_multiplier = 2
train_with_crops = False


''' Rats fold 1-3 '''
for i in range(2,11):
    for k in range(1,4):
        ratnum = 'Rat' + str(i)
        for dense_neurons in dense_neurons_arr:
            for numfilters in numfilters_arr:
                for filtsize in filtsizes:
                    for num_layer in num_layers:
                        CNN.runCNN_full(ratnum,k,epochs,batch_size,valid_patience,min_delta,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,print_model,load_se_te,load_combined,load_student,stack_conv,channel_width_multiplier,train_with_crops,num_layer)

       
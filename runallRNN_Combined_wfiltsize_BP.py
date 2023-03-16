# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:00:22 2018

@author: kohr
"""

import RNN_new_combined_wfiltsize_BP as RNN
from File_utils import File_utils 

NN_hyperparameters = dict()
directives = dict()

valid_patience = 15
min_delta = 0
restore_best_weights = False
epochs = 1000
batch_size = 50
numspikes = 1000
filter_dense_numlayers_arr = [[32,16,3]]
layer_type_enum_arr = [File_utils.LayerType.LSTM]
activation_enum_arr = [None]
filtsizes = [8]
dropout_rates = [0]
channel_width_multiplier = 1
use_attention = [True]
print_model = True
load_se_te = False
load_combined = False


''' Rats fold 1-3 '''
for i in range(2,11):
    for k in range(1,4):
        ratnum = 'Rat' + str(i)
        
        for filter_dense_numlayers in filter_dense_numlayers_arr:
            numfilters = filter_dense_numlayers[0]
            dense_neurons = filter_dense_numlayers[1]
            num_layers = filter_dense_numlayers[2]

            for filtsize in filtsizes:
                for dropout_rate in dropout_rates:
                    for layer_type_enum in layer_type_enum_arr:
                        for activation_enum in activation_enum_arr:
                            for use_attn in use_attention:
                        
                                NN_hyperparameters.clear()
                                directives.clear()
                                NN_hyperparameters["foldnum"] = k
                                NN_hyperparameters["epochs"] = epochs
                                NN_hyperparameters["batch_size"] = batch_size
                                NN_hyperparameters["valid_patience"] = valid_patience
                                NN_hyperparameters["min_delta"] = min_delta
                                NN_hyperparameters["restore_best_weights"] = restore_best_weights
                                NN_hyperparameters["numspikes"] = numspikes
                                NN_hyperparameters["numfilters"] = numfilters
                                NN_hyperparameters["filtsize"] = filtsize
                                NN_hyperparameters["dropout_rate"] = dropout_rate
                                NN_hyperparameters["dense_neurons"] = dense_neurons
                                NN_hyperparameters["num_layers"] = num_layers
                                NN_hyperparameters["layer_type_enum"] = layer_type_enum
                                NN_hyperparameters["activation_enum"] = activation_enum
                                NN_hyperparameters["channel_width_multiplier"] = channel_width_multiplier
                                NN_hyperparameters["use_attention"] = use_attn
                                directives["ratnum"] = ratnum
                                directives["print_model"] = print_model
                                directives["load_se_te"] = load_se_te
                                directives["load_combined"] = load_combined
                                RNN.runRNN_full(NN_hyperparameters,directives)
                                
                                
                                                 
                                                 
                                                 
                                                 

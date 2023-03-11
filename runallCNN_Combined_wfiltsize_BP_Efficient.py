# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:00:22 2018

@author: kohr
"""

import CNN_new_combined_wfiltsize_BP_Efficient as CNN

valid_patience = 15
epochs = 1000
batch_size = 50
numspikes = 1000
numfilters_arr = [32]#16,32]#32]##16,32]#32]
dense_neurons_arr = [32]#16,32,64]#16,32,64]#32,64,16]#8,4]
#filtsize = 8
filtsizes = [8]#8]#9]#9]
dropout_rates = [0]#0]#0.5]#0.5
channel_width_multiplier = 1
num_layers = [4]
# filepath = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Ryan K\\CNNs\\Training_Sets_BP\\'
print_model = True 
                 

# for combo in manual_combos:
#     i = combo[0]
#     numfilters = combo[1]
#     k = combo[2]
#     dense_neurons = combo[3]
#     dropout_rate = combo[4]
#     ratnum = 'Rat' + str(i)
#     num_layer = combo[5]
#     filtsize = combo[6]
''' Rats fold 1-3 '''
for i in range(71,72):
    for k in range(1,4):
        # ERat FOR NEW DATASET COLLECTED BY EUGENE
        ratnum = 'ERat' + str(i)
        
        for dense_neurons in dense_neurons_arr:
            for numfilters in numfilters_arr:
                for filtsize in filtsizes:
                    for dropout_rate in dropout_rates:
                        for num_layer in num_layers:
                            CNN.runCNN_full(ratnum,k,epochs,batch_size,valid_patience,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,channel_width_multiplier,num_layer,print_model)

               
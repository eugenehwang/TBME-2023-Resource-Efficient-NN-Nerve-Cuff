# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:24:24 2018

@author: kohr
"""

import numpy as np 
import scipy.io
import random
from File_utils import File_utils
import matplotlib.pyplot as plt


class Preprocessing_module:
    
    '''Constructor'''    
    def __init__(self,Ratnum,foldnum,is_RNN=False):
        
        load_filename = 'D:\\Eugene\\Training_Sets_ConPerRing\\' + Ratnum + 'Training_Fold' + str(foldnum)
        
        self.numcons = 56
        
        RAT = scipy.io.loadmat(load_filename)
        
        training_data = RAT['training_data_rat'];
        test_data = RAT['test_data_rat'];
        
        training_data = training_data.T;
        test_data = test_data.T;
        
        if is_RNN:
            test_data = test_data.reshape(test_data.shape[0],100,self.numcons,1)
            training_data = training_data.reshape(training_data.shape[0],100,self.numcons,1)
        else:
            test_data = test_data.reshape(test_data.shape[0],self.numcons,100,1)
            training_data = training_data.reshape(training_data.shape[0],self.numcons,100,1)
        
        self.training_set = training_data
        self.training_labels = RAT['training_data_labels']
        self.test_set = test_data
        self.test_labels = RAT['test_data_labels']
        self.valid_set = []
        self.valid_labels = []
        self.samples1 = []
        self.samples2 = []
        self.samples3 = []
        
    def clear(self):
        self.training_set = []
        self.training_labels = []
        self.test_set = []
        self.test_labels = []
        self.valid_set = []
        self.valid_labels = []
        self.samples1 = []
        self.samples2 = []
        self.samples3 = []
        
    def update(self,RAT_data_int):
        if self.training_set == []:
            self.training_set = RAT_data_int.training_set
        else:
            self.training_set = np.append(self.training_set,RAT_data_int.training_set, axis=0)
        
        if self.training_labels == []:
            self.training_labels = RAT_data_int.training_labels
        else:
            self.training_labels = np.append(self.training_labels,RAT_data_int.training_labels, axis=0)
        
        if self.valid_set == []:
            self.valid_set = RAT_data_int.valid_set
        else:
            self.valid_set = np.append(self.valid_set,RAT_data_int.valid_set, axis=0)
            
        if self.valid_labels == []:
            self.valid_labels = RAT_data_int.valid_labels
        else:
            self.valid_labels = np.append(self.valid_labels,RAT_data_int.valid_labels, axis=0)
        
        if self.samples1 == [] and self.samples2 == [] and self.samples3 == []:
            self.samples1 = RAT_data_int.samples1
            self.samples2 = RAT_data_int.samples2
            self.samples3 = RAT_data_int.samples3
        else:
            self.samples1 = np.append(self.samples1,RAT_data_int.samples1, axis=0)
            self.samples2 = np.append(self.samples2,RAT_data_int.samples2, axis=0)
            self.samples3 = np.append(self.samples3,RAT_data_int.samples3, axis=0)
        
    def update_test(self,RAT_data_int):
        if self.test_set == []:
            self.test_set = RAT_data_int.test_set
            self.test_labels = RAT_data_int.test_labels
        else:
            self.test_set = np.append(self.test_set,RAT_data_int.test_set, axis=0)
            self.test_labels = np.append(self.test_labels,RAT_data_int.test_labels, axis=0)
                
    def Reshape_data_set(self,training_data,test_data):
        
        self.test_set = test_data.reshape(test_data.shape[0],self.numcons,100,1)
        self.training_set = training_data.reshape(training_data.shape[0],self.numcons,100,1)
        
    def Reshape_data_set_RNN(self):
        
        self.test_set = self.test_set.reshape(self.test_set.shape[0],100,self.numcons)
        self.training_set = self.training_set.reshape(self.training_set.shape[0],100,self.numcons)
        if len(self.valid_set) > 0:
            self.valid_set = self.valid_set.reshape(self.valid_set.shape[0],100,self.numcons)
        
    def Randomize_Train_and_getValidSet(self,training_data,training_labels,numspikes):
        random.seed(10)
        itemindex = np.where(training_labels==1)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples1 = temp
        validation_set = training_data[itemindex[0][temp],:,:,:]
        validation_labels =  training_labels[itemindex[0][temp]]
        indices = itemindex[0][temp]
        itemindex = np.where(training_labels==2)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples2 = temp
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        itemindex = np.where(training_labels==3)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples3 = temp
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        
        training_data = np.delete(training_data,indices,axis = 0)
        training_labels = np.delete(training_labels,indices,axis = 0)
        
        self.valid_set = validation_set
        self.valid_labels = validation_labels
        self.training_set = training_data
        self.training_labels = training_labels
        
    def Randomize_Train_and_getValidSet2(self,training_data,training_labels,sample1,sample2,sample3,numspikes):
        itemindex = np.where(training_labels==1)
        temp = sample1
        validation_set = training_data[itemindex[0][temp],:,:,:]
        validation_labels =  training_labels[itemindex[0][temp]]
        indices = itemindex[0][temp]
        itemindex = np.where(training_labels==2)
        temp = sample2
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        itemindex = np.where(training_labels==3)
        temp = sample3
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        
        training_data = np.delete(training_data,indices,axis = 0)
        training_labels = np.delete(training_labels,indices,axis = 0)
        
        self.valid_set = validation_set
        self.valid_labels = validation_labels
        self.training_set = training_data
        self.training_labels = training_labels
        
        
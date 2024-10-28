#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:45:43 2023

@author: pritcham

Script for feature-extracting ANY dataset, from a folder of raw biosignal recordings to a single CSV
also functionality to split a featureset into individual Holdout subjects & a "remainder" Development set
"""


import params as params
import handleFeats as feats
import pandas as pd


def split_holdout(full_path,holdoutIDs,save_path_stem):
    fullset = pd.read_csv(full_path,delimiter=',')
    
    for HO in holdoutIDs:
        HOset = fullset[fullset['ID_pptID']==HO]
        HOset.to_csv((save_path_stem+'_ppt'+str(HO)+'.csv'),
                     sep=',',index=False,float_format="%.18e")
        
    devset = fullset[~fullset['ID_pptID'].isin(holdoutIDs)]
    devset.to_csv((save_path_stem+'_noHO.csv'),
                  sep=',',index=False,float_format="%.18e")
    

def get_paths(dataset):

    if dataset=='jeongEMG':           # jeong EMG
        directory_path=params.jeong_EMGdir
        output_file=params.jeong_EMGfeats
        
    elif dataset=='jeongEEG':    #jeong EEG
        directory_path=params.jeong_EEGdir
        output_file=params.jeong_EEGfeats
        
    else:
         raise ValueError('I don\'t know what dataset you mean by '+dataset)
         
    return directory_path,output_file


def extract_featureset(directory_path, output_file, skipfails=True, period=1000, datatype='eeg'):
    print('Are the following parameters OK?')
    print('Skipfails: ',skipfails,'\n',
          'Period: ',period,'\n',
          'datatype: ',datatype,'\n',
          'Data directory: ',directory_path,'\n',
          'Dataset location: ',output_file)
    
    go_ahead=input('Ready to proceed [Y/N] ')
    
    if go_ahead=='Y' or go_ahead=='y':
        feats.make_feats(directory_path, output_file, datatype, period, skipfails)
    else:
        print('aborting...')
    
    
if __name__ == '__main__':

    dataset='jeongEMG'
    
    skipfails=True
    period=1000
    datatype='emg'
    holdoutIDs=[1,6,11,16,21]
    
    directory_path,output_file = get_paths(dataset)
    
    extract_featureset(directory_path, output_file, skipfails, period, datatype)
    
    full_path = output_file
    save_path_stem = output_file[:-4]
    split_holdout(full_path,holdoutIDs,save_path_stem)
    
    ##########
    
    dataset='jeongEEG'
    
    skipfails=True
    period=1000
    datatype='eeg'
    holdoutIDs=[1,6,11,16,21]
    
    directory_path,output_file = get_paths(dataset)
    
    extract_featureset(directory_path, output_file, skipfails, period, datatype)
    
    full_path = output_file
    save_path_stem = output_file[:-4]
    split_holdout(full_path,holdoutIDs,save_path_stem)


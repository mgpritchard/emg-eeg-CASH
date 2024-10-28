#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:15:27 2022

@author: pritcham

module to contain functionality related to feature engineering (extraction, selection, and scaling)

"""
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
import generate_training_matrix as genfeats
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np


def sel_percent_feats_df(data,percent=15):
    target=data['Label']
    attribs=data.drop(columns=['Label'])
    selector=SelectPercentile(f_classif,percentile=percent)
    selector.fit(attribs,target)
    col_idxs=selector.get_support(indices=True)
    #selected=attribs.iloc[:col_idxs]
    #selected['Label']=target
    return col_idxs

def sel_feats_l1_df(data,sparsityC=0.01,maxfeats=None):
    target=data['Label']
    attribs=data.drop(columns=['Label'])   
    lsvc = LinearSVC(C=sparsityC, penalty="l1", dual=False).fit(attribs, target)
    if maxfeats is None:
        #getting all nonzero feats
        model = SelectFromModel(lsvc, prefit=True)
    else:
        #getting feats up to maxfeats, with no threshold of rating so as to ensure never <maxfeats
        model = SelectFromModel(lsvc, prefit=True,threshold=-np.inf,max_features=maxfeats)   
    col_idxs=model.get_support(indices=True)
    return col_idxs

def scale_feats_train(data,mode='normalise'):
    '''data is a dataframe of feats, mode = normalise or standardise'''
    if mode is None:
        return data, None
    if mode=='normalise' or mode=='normalize':
        scaler=Normalizer()
    elif mode=='standardise' or mode=='standardize':
        scaler=StandardScaler()
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    return data, scaler

def scale_feats_test(data,scaler):
    '''data is a dataframe of feats, scaler is a scaler fit to training data'''
    if scaler is None:
        return data
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    return data    

def ask_for_dir(datatype=""):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='directory of '+datatype+' data for feature extraction' 
    set_dir=askdirectory(title=title,initialdir=homepath)
    return set_dir

def ask_for_savefile(datatype=""):
    homepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Tk().withdraw()
    title='save '+datatype+' featureset as' 
    savefile=asksaveasfilename(title=title,initialdir=homepath)
    return savefile

def make_feats(directory_path=None, output_file=None, datatype="",period=1000,skipfails=False):
    '''datatype can be in any case, it does not affect mechanical functionality only cosmetic'''
    if directory_path is None:
        directory_path=ask_for_dir(datatype)
    if output_file is None:
        output_file=ask_for_savefile(datatype)
    featset=genfeats.gen_training_matrix(directory_path, output_file, cols_to_ignore=None, singleFrame=0,period=period,auto_skip_all_fails=skipfails)
    return featset
    
    
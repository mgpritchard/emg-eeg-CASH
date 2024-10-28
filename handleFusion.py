#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:20:45 2021

@author: pritcham

module to contain functionality related to [decision-level] fusion
"""

import os, sys
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import csv
import params
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import handleML as ml
import random

def setup_onehot(classlabels):
    labels=[params.idx_to_gestures[label] for label in classlabels]
    options=np.array(labels).reshape(-1,1)
    ohe_dense=OneHotEncoder(sparse=False)
    ohe_dense.fit_transform(options)
    return ohe_dense    
    
def encode_preds_onehot(preds,encoder):
    preds_labelled=np.array([params.idx_to_gestures[pred] for pred in preds],dtype=object).reshape(-1,1)
    output=encoder.transform(preds_labelled)
    return output


def train_svm_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    C=args['svm_C']
    if 'fusesvmPlatt' in args:
        if args['fusesvmPlatt']==True:
            kernel=args['kernel']
            gamma=args['gamma']
            if kernel=='linear':
                model=ml.SVC(C=C,kernel=kernel,probability=True)
            else:
                model=ml.SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
        else:
            model=ml.LinearSVC(C=C,dual=False)
    else:        
        model=ml.LinearSVC(C=C,dual=False)
    model.fit(train.astype(np.float64),targets)
    return model

def svm_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def train_lda_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    solver=args['LDA_solver']
    shrinkage=args['shrinkage']
    if solver == 'svd':
        model=LinearDiscriminantAnalysis(solver=solver)
    else:
        model=LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
    model.fit(train.astype(np.float64),targets)
    return model

def lda_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds

def train_rf_fuser(mode1,mode2,targets,args):
    train=np.column_stack([mode1,mode2])
    n_trees=args['n_trees']
    max_depth=args['max_depth']
    model=ml.RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth)
    model.fit(train.astype(np.float64),targets)
    return model

def rf_fusion(fuser,onehot,predlist_emg,predlist_eeg,classlabels):
    if onehot is not None:
        predlist_emg=encode_preds_onehot(predlist_emg,onehot)
        predlist_eeg=encode_preds_onehot(predlist_eeg,onehot)
    fusion_preds=fuser.predict(np.column_stack([predlist_emg,predlist_eeg]))
    return fusion_preds


def normalise_weights(w1,w2):
    wtot = w1+w2
    w1 = w1 / wtot
    w2 = w2 / wtot
    return w1,w2

def fuse_max(mode1,mode2):
    if max(mode1)>=max(mode2):
        fused=mode1
    else:
        fused=mode2
    return fused

def fuse_max_arr(l1, l2):
    return np.array([a if max(a)>max(b) else random.choice([a,b]) if max(a)==max(b) else b for a, b in zip(l1, l2)])

def fuse_conf(mode1,mode2):
    fused=[]
    for instance in range(len(mode1)):
        fused.append(fuse_max(mode1[instance,:],mode2[instance,:]))
    return fused

def fuse_linweight(mode1,mode2,weight1,weight2):
    weight1,weight2=normalise_weights(weight1, weight2)
    fused=(mode1*weight1)+(mode2*weight2)
    return np.asarray(fused)

def fuse_select(emg,eeg,args):
    alg=args['fusion_alg']
    if type(alg) is dict:
        alg=alg['fusion_alg_type']
    if alg=='mean':
        fusion = fuse_mean(emg,eeg)
    elif alg=='3_1_emg':
        fusion = fuse_linweight(emg,eeg,75,25)
    elif alg=='3_1_eeg':
        fusion = fuse_linweight(emg,eeg,25,75)
    elif alg=='opt_weight':
        fusion = fuse_linweight(emg,eeg,100-args['eeg_weight_opt'],args['eeg_weight_opt'])
    elif alg=='highest_conf':
        fusion = fuse_max_arr(emg,eeg)
    elif alg=='featlevel':
        '''feature level fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='svm':
        '''SVM fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='lda':
        '''LDA fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    elif alg=='rf':
        '''RF fusion is not done here, just keeping system happy'''
        fusion = fuse_mean(emg,eeg)
    else:
        msg='Fusion algorithm '+alg+' not recognised'
        raise NotImplementedError(msg)
    return fusion

def fuse_mean(mode1,mode2):
    mean=fuse_linweight(mode1,mode2,50,50)
    return mean
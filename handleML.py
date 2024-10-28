#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:37:59 2021

@author: pritcham

module to contain functionality related to ML classification
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sklearn as skl
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale, label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import types


import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
import time

def eval_acc(results,state):
    count=results.count(state)
    acc=count/len(results)
    return acc

def eval_fusion(mode1,mode2,fusion,state):
    acc1=eval_acc(mode1,state)
    acc2=eval_acc(mode2,state)
    accF=eval_acc(fusion,state)
    return [acc1,acc2,accF]

def eval_multi(results,state):
    acc=[]
    for result in results:
        acc.append(eval_acc(result,state))
    return acc

def load_model(name,path):
    title='select saved model for '+name
    Tk().withdraw()
    model_name=askopenfilename(initialdir=path,title=title,filetypes = (("sav files","*.sav"),("all files","*.*")))
    with open(model_name,'rb') as load_model:
        model = pickle.load(load_model)
    return model

def matrix_from_csv_file(file):
    csv_data=pd.read_csv(file,delimiter=",").values
    matrix = csv_data[1:]
    headers = csv_data[0]
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def drop_ID_cols(csv_dframe):
    IDs=csv_dframe.filter(regex='^ID_').columns
    csv_dframe=csv_dframe.drop(IDs,axis='columns')
    '''may benefit from a trycatch in case of keyerror?'''
    return csv_dframe

def matrix_from_csv_file_drop_ID(file):
    csv_dframe=pd.read_csv(file,delimiter=",")
    csv_dframe=drop_ID_cols(csv_dframe)
    matrix=csv_dframe.values
    headers = csv_dframe.columns.values
    print ('MAT', (matrix.shape))
	#print ('HDR', (headers.shape))
    return matrix, headers

def train_optimise(training_set,modeltype,args):
    '''where training_set is a Pandas dataframe
    which has Label as the last column but has had ID columns dropped'''
    if modeltype=='RF':
        model=train_RF_param(training_set,args)
    elif modeltype=='gaussNB':
        model=train_gnb(training_set,args)
    elif modeltype=='LDA':
        model=train_LDA_param(training_set,args)
    elif modeltype=='kNN':
        model = train_knn(training_set,args)
    elif modeltype=='SVM':
        raise ValueError('LinearSVC has no predict_proba')
    elif modeltype=='QDA':
        model = train_QDA(training_set,args)
    elif modeltype=='SVM_PlattScale':
        model = train_SVC_Platt(training_set,args)
    elif modeltype=='LR':
        model=train_LR(training_set,args)
   
    return model

def train_LR(train_data,args):
    C=args['C']
    solver=args['solver']
    model=LogisticRegression(C=C,solver=solver)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_gnb(train_data,args):
    smoothing=args['smoothing']
    model=GaussianNB(var_smoothing=smoothing)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_SVC_Platt(train_data,args):
    kernel=args['kernel']
    C=args['svm_C']
    gamma=args['gamma']
    if kernel=='linear':
        model=SVC(C=C,kernel=kernel,probability=True)
    else:
        svc=SVC(C=C,kernel=kernel,gamma=gamma)
        model=CalibratedClassifierCV(svc,cv=5)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_QDA(train_data,args):
    reg=args['regularisation']
    model=QuadraticDiscriminantAnalysis(reg_param=reg)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model


def train_LDA_param(train_data,args):
    solver=args['LDA_solver']
    shrinkage=args['shrinkage']
    if solver == 'svd':
        model=LinearDiscriminantAnalysis(solver=solver)
    else:
        model=LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_knn(train_data,args):
    k=args['knn_k']
    model=KNeighborsClassifier(n_neighbors=k)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_RF_param(train_data,args):
    '''where args is a dictionary with n_trees as an integer item within'''
    n_trees=args['n_trees']
    max_depth=args['max_depth']
    model=RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model
    
def train_model(model,data):
    train=data[:,:-1]
    targets=data[:,-1]
    train1,train2,test1=np.array_split(train,3)
    train1=train1.astype(np.float64)
    train2=train2.astype(np.float64)
    traindat=np.concatenate((train1,train2))
    test1=test1.astype(np.float64)
    targets1,targets2,targetstest=np.array_split(targets,3)
    targetsdat=np.concatenate((targets1,targets2))
    model.fit(traindat,targetsdat)
    results=model.predict(test1)
    acc=accuracy_score(targetstest,results)
    model.fit(train.astype(np.float64),targets)
    return model, acc

def prob_dist(model,values):
    distro = model.predict_proba(values)
    distro[distro==0]=0.00001
    return distro

def predict_from_array(model,values):
	prediction = model.predict(values)
	return prediction

def pred_from_distro(labels,distro):
    pred=int(np.argmax(distro))
    label=labels[pred]
    return label

def predlist_from_distrosarr(labels,distros):
    predcols=np.argmax(distros,axis=1)
    predlabels=labels[predcols]
    return predlabels.tolist()

def pred_gesture(prediction,toggle_print):

    if isinstance(prediction,int):
        if prediction == 2: #typically 0
            gesture='open'
        elif prediction == 1:
            gesture='neutral'
        elif prediction == 0: #typically 2
            gesture='close'
    else:
        gesture=prediction
        
    if toggle_print:
        print(time.time())
        print(gesture)
        print('-------------')

    return gesture  

def classify_instance(frame,model):
    prediction=frame
    return prediction

def classify_continuous(data):
    while True:
        pred=classify_instance(data,'NB')
        yield pred
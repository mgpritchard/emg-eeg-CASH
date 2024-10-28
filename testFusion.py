#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:42:00 2022

@author: pritcham
"""

import os
import sys
import numpy as np
import statistics as stats
import handleDatawrangle as wrangle
import handleFeats as feats
import handleML as ml
import handleComposeDataset as comp
import handleTrainTestPipeline as tt
import handleFusion as fusion
from handlePlotResults import plot_opt_in_time, plot_stat_in_time, plot_stat_as_line, plot_multiple_stats, calc_runningbest, plot_multiple_stats_with_best, plot_multi_runbest_df, boxplot_param, scatterbox
import params
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope, stochastic
import time
import pandas as pd
import pickle as pickle


def inspect_set_balance(emg_set_path=None,eeg_set_path=None,emg_set=None,eeg_set=None):
    if emg_set is None:
        if emg_set_path is None:
            raise ValueError
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
    emg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    if emg_set_path:
        print(emg_set_path.split('/')[-1])
    else:
        print('EMG:')
    print(emg_set['Label'].value_counts())
    print(emg_set['ID_pptID'].value_counts())
    
    if eeg_set is None:
        if eeg_set_path is None:
            raise ValueError
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    eeg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    if eeg_set_path:
        print(eeg_set_path.split('/')[-1])
    else:
        print('EEG:')
    print(eeg_set['Label'].value_counts())
    print(eeg_set['ID_pptID'].value_counts())
    
    return emg_set,eeg_set

def balance_single_mode(dataset):
    dataset['ID_stratID']=dataset['ID_pptID'].astype(str)+dataset['Label'].astype(str)
    stratsize=np.min(dataset['ID_stratID'].value_counts())
    balanced = dataset.groupby('ID_stratID')
    balanced=balanced.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
    return balanced

def balance_set(emg_set,eeg_set):
    #print('initial')
    #_,_=inspect_set_balance(emg_set=emg_set,eeg_set=eeg_set)
    emg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_set[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=emg_set.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=eeg_set.loc[index_eeg.isin(index_emg)].reset_index(drop=True)    
    
    eeg['ID_stratID']=eeg['ID_pptID'].astype(str)+eeg['Label'].astype(str)
    emg['ID_stratID']=emg['ID_pptID'].astype(str)+emg['Label'].astype(str)
    
    stratsize=np.min(emg['ID_stratID'].value_counts())
    balemg = emg.groupby('ID_stratID',group_keys=False)
    balemg=balemg.apply(lambda x: x.sample(stratsize))
    print('subsampling to ',str(stratsize),' per combo of ppt and class')
   
    #print('----------\nEMG Balanced')
    #_,_=inspect_set_balance(emg_set=balemg,eeg_set=eeg)
    
    index_balemg=ml.pd.MultiIndex.from_arrays([balemg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    baleeg=eeg_set.loc[index_eeg.isin(index_balemg)].reset_index(drop=True)
   
    #print('----------\nBoth Balanced')
    #_,_=inspect_set_balance(emg_set=balemg,eeg_set=baleeg)
    
    if 0:   #manual checking, almost certainly unnecessary
        for index,emgrow in balemg.iterrows():
            eegrow = baleeg[(baleeg['ID_pptID']==emgrow['ID_pptID'])
                                  & (baleeg['ID_run']==emgrow['ID_run'])
                                  & (baleeg['Label']==emgrow['Label'])
                                  & (baleeg['ID_gestrep']==emgrow['ID_gestrep'])
                                  & (baleeg['ID_tend']==emgrow['ID_tend'])]
            #syntax like the below would do it closer to a .where
            #eegrow=test_set_eeg[test_set_eeg[['ID_pptID','Label']]==emgrow[['ID_pptID','Label']]]
            if eegrow.empty:
                print('No matching EEG window for EMG window '+str(emgrow['ID_pptID'])+str(emgrow['ID_run'])+str(emgrow['Label'])+str(emgrow['ID_gestrep'])+str(emgrow['ID_tend']))
                continue
            
            TargetLabel=emgrow['Label']
            if TargetLabel != eegrow['Label'].values:
                raise Exception('Sense check failed, target label should agree between modes')
        print('checked all for window matching')
    
    balemg.drop(columns='ID_stratID',inplace=True)
    return balemg,baleeg


def get_ppt_split(featset,args={'using_literature_data':True},mask_dict=None):
    if args['using_literature_data']:
        masks=[featset['ID_pptID']== n_ppt for n_ppt in np.sort(featset['ID_pptID'].unique())] 
        return masks
    else:
        raise ValueError('Specify participant masks yourself if using primary data')

def isolate_holdout_ppts(ppts):
    emg_set=ml.pd.read_csv(params.jeong_EMGfeats,delimiter=',')
    eeg_set=ml.pd.read_csv(params.jeong_noCSP_WidebandFeats,delimiter=',')
    emg_set,eeg_set=balance_set(emg_set,eeg_set)
    emg_masks = get_ppt_split(emg_set)
    eeg_masks = get_ppt_split(eeg_set)
    for idx, emg_mask in enumerate(emg_masks):
        if idx not in ppts:
            continue
        else:
            eeg_mask=eeg_masks[idx]
            emg=emg_set[emg_mask]
            eeg=eeg_set[eeg_mask]
            emg_set.drop(emg_set[emg_mask].index,inplace=True)
            eeg_set.drop(eeg_set[eeg_mask].index,inplace=True)
            emg.to_csv((r"H:\Jeong11tasks_data\final_dataset\holdout\emg_holdout_ppt"+str(idx+1)+'.csv'),index=False)
            eeg.to_csv((r"H:\Jeong11tasks_data\final_dataset\holdout\eeg_holdout_ppt"+str(idx+1)+'.csv'),index=False)
    inspect_set_balance(emg_set=emg_set,eeg_set=eeg_set)
    emg_set.to_csv((r"H:\Jeong11tasks_data\final_dataset\emg_set_noholdout.csv"),index=False)
    eeg_set.to_csv((r"H:\Jeong11tasks_data\final_dataset\eeg_set_noholdout.csv"),index=False)


def make_featsel_all20(joint=True):
    '''feat array for generalist trained on all 20 dev ppts'''
    args=setup_search_space('featlevel',False)
    emg_others=ml.pd.read_csv(args['emg_set_path'],delimiter=',')
    eeg_others=ml.pd.read_csv(args['eeg_set_path'],delimiter=',')
    emg_others,eeg_others=balance_set(emg_others,eeg_others)
      
    emg_others,_=feats.scale_feats_train(emg_others,args['scalingtype'])
    eeg_others,_=feats.scale_feats_train(eeg_others,args['scalingtype'])
  
    args.update({'emg_set':emg_others,'eeg_set':eeg_others,'data_in_memory':True,'prebalanced':True})
    args.update({'l1_sparsity':0.005})
    args.update({'l1_maxfeats':88}) 
    
  #  emg_others = emg_set
  #  eeg_others = eeg_set
   
    emg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_others.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_others[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_others=emg_others.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_others=eeg_others.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    if emg_others['Label'].equals(eeg_others['Label']):
        pass
    else:
        raise RuntimeError('Target classes should match, training sets are misaligned')
        
    eeg_others=ml.drop_ID_cols(eeg_others)
    emg_others=ml.drop_ID_cols(emg_others)
    
    if joint:    
        eeg_others.drop('Label',axis='columns',inplace=True)
        eeg_others.rename(columns=lambda x: 'EEG_'+x, inplace=True)
        labelcol=emg_others.pop('Label')
        emgeeg_others=pd.concat([emg_others,eeg_others],axis=1)
        emgeeg_others['Label']=labelcol
        
        sel_cols_emgeeg=feats.sel_feats_l1_df(emgeeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats']+88)
        '''here we are taking total features = N(EEG feats) + N(EMG feats) = N(EEG) + 88'''
        sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_others.columns.get_loc('Label'))
    
        emgeeg_others = emgeeg_others.iloc[:,sel_cols_emgeeg] 
        
        joincols=emgeeg_others.columns.values
        join_feats_df=pd.DataFrame(joincols)
        
        join_feats_df.to_csv(params.joint_LOO_HOtest_feats_csv,index=False,header=False)
    else:
        sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))
        
        sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
        
        emg_others = emg_others.iloc[:,sel_cols_emg] 
        eeg_others = eeg_others.iloc[:,sel_cols_eeg]
        
        emgcols=emg_others.columns.values
        eegcols=eeg_others.columns.values
        eeg_feats_df=pd.DataFrame(eegcols)
        emg_feats_df=pd.DataFrame(emgcols)
        eeg_feats_df.to_csv(params.eeg_LOO_HOtest_feats_csv,index=False,header=False)
        emg_feats_df.to_csv(params.emg_LOO_HOtest_feats_csv,index=False,header=False)


def refactor_synced_predict(test_set_emg,test_set_eeg,model_emg,model_eeg,classlabels,args, chosencolseeg=None, chosencolsemg=None, get_distros=False):

    index_emg=ml.pd.MultiIndex.from_arrays([test_set_emg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([test_set_eeg[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg=test_set_emg.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg=test_set_eeg.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    predlist_emg, predlist_eeg, predlist_fusion, targets = [], [], [], []
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')
        
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    eeg=eeg.drop(IDs,axis='columns')
    
    if chosencolseeg is not None:
        eeg=eeg.iloc[:,chosencolseeg]
    if chosencolsemg is not None:
        emg=emg.iloc[:,chosencolsemg]
    emgvals=emg.drop(['Label'],axis='columns').values
    eegvals=eeg.drop(['Label'],axis='columns').values
    
    '''Pass values to models'''
    distros_emg=ml.prob_dist(model_emg,emgvals)
    predlist_emg=ml.predlist_from_distrosarr(classlabels,distros_emg)

    distros_eeg=ml.prob_dist(model_eeg,eegvals)
    predlist_eeg=ml.predlist_from_distrosarr(classlabels,distros_eeg)

    distros_fusion=fusion.fuse_select(distros_emg, distros_eeg, args)
    predlist_fusion=ml.predlist_from_distrosarr(classlabels,distros_fusion)
    
    if get_distros:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, distros_emg, distros_eeg, distros_fusion
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, None, None, None


def classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels):
    '''Convert predictions to gesture labels'''
    gest_truth=[params.idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[params.idx_to_gestures[pred] for pred in predlist_emg]
    gest_pred_eeg=[params.idx_to_gestures[pred] for pred in predlist_eeg]
    gest_pred_fusion=[params.idx_to_gestures[pred] for pred in predlist_fusion]
    gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
    
    return gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels

def plot_confmats(gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels):
    '''Produce confusion matrix'''
    tt.confmat(gest_truth,gest_pred_emg,gesturelabels)
    tt.confmat(gest_truth,gest_pred_eeg,gesturelabels)
    tt.confmat(gest_truth,gest_pred_fusion,gesturelabels)

def train_models_opt(emg_train_set,eeg_train_set,args):
    emg_model_type=args['emg']['emg_model_type']
    eeg_model_type=args['eeg']['eeg_model_type']
    emg_model = ml.train_optimise(emg_train_set, emg_model_type, args['emg'])
    eeg_model = ml.train_optimise(eeg_train_set, eeg_model_type, args['eeg'])
    return emg_model,eeg_model

def train_metamodel_fuser(model_emg,model_eeg,emg_set,eeg_set,classlabels,args,sel_cols_eeg,sel_cols_emg):
    targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_set, eeg_set, model_emg, model_eeg, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    if not args['stack_distros']:
        onehot=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        if args['fusion_alg']=='svm':
            fuser=fusion.train_svm_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['svmfuse'])
        elif args['fusion_alg']=='lda':
            fuser=fusion.train_lda_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['ldafuse'])
        elif args['fusion_alg']=='rf':
            fuser=fusion.train_rf_fuser(onehot_pred_emg, onehot_pred_eeg, targets, args['RFfuse'])
    else:
        onehot=None
        if args['fusion_alg']=='svm':
            fuser=fusion.train_svm_fuser(distros_emg, distros_eeg, targets, args['svmfuse'])
        elif args['fusion_alg']=='lda':
            fuser=fusion.train_lda_fuser(distros_emg, distros_eeg, targets, args['ldafuse'])
        elif args['fusion_alg']=='rf':
            fuser=fusion.train_rf_fuser(distros_emg, distros_eeg, targets, args['RFfuse'])
    return fuser, onehot

def sort_harmonise_data(emg_in,eeg_in):
    emg_in.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_in.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    
    index_emg=ml.pd.MultiIndex.from_arrays([emg_in[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    index_eeg=ml.pd.MultiIndex.from_arrays([eeg_in[col] for col in ['ID_pptID','ID_run','Label','ID_gestrep','ID_tend']])
    emg_out=emg_in.loc[index_emg.isin(index_eeg)].reset_index(drop=True)
    eeg_out=eeg_in.loc[index_eeg.isin(index_emg)].reset_index(drop=True)
    
    return emg_out, eeg_out

def feature_fusion(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''    
    emg_others, eeg_others = sort_harmonise_data(emg_others, eeg_others)
    
    if emg_others['Label'].equals(eeg_others['Label']):
        #print('Target classes match, ok to merge sets')
        pass
    else:
        raise RuntimeError('Target classes should match, training sets are misaligned')
    
     
    eeg_others=ml.drop_ID_cols(eeg_others)
    emg_others=ml.drop_ID_cols(emg_others)
    
    if not args['featfuse_sel_feats_together']:
        
        if args['trialmode']=='WithinPpt':
            sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))

            sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
        elif args['trialmode']=='LOO':
            idx=int(emg_ppt['ID_pptID'].iloc[0])-1
            sel_cols_emg=[emg_others.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
            sel_cols_eeg=[eeg_others.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
            
        emg_others = emg_others.iloc[:,sel_cols_emg] 
        eeg_others = eeg_others.iloc[:,sel_cols_eeg]


    eeg_others.drop('Label',axis='columns',inplace=True)
    eeg_others.rename(columns=lambda x: 'EEG_'+x, inplace=True)
    labelcol=emg_others.pop('Label')
    emgeeg_others=pd.concat([emg_others,eeg_others],axis=1)
    emgeeg_others['Label']=labelcol
    
    if args['featfuse_sel_feats_together']:
        if args['trialmode']=='WithinPpt':
            sel_cols_emgeeg=feats.sel_feats_l1_df(emgeeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats']+88)
            '''here we are taking total features = N(EEG feats) + N(EMG feats) = N(EEG) + 88'''
            sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_others.columns.get_loc('Label'))
        elif args['trialmode']=='LOO':
            idx=int(emg_ppt['ID_pptID'].iloc[0])-1
            sel_cols_emgeeg=[emgeeg_others.columns.get_loc(col) for col in args['jointemgeeg_feats_LOO'].iloc[idx].tolist()]

        emgeeg_others = emgeeg_others.iloc[:,sel_cols_emgeeg]
    
    emgeeg_model = ml.train_optimise(emgeeg_others, args['featfuse']['featfuse_model_type'],args['featfuse'])
    
    classlabels = emgeeg_model.classes_
    
    '''TESTING ON PPT DATA'''
    emg_ppt, eeg_ppt = sort_harmonise_data(emg_ppt,eeg_ppt)
    
    if emg_ppt['Label'].equals(eeg_ppt['Label']):
        #print('Target classes match, ok to merge sets')
        targets=emg_ppt['Label'].values.tolist()
    else:
        raise RuntimeError('Sense check failed, target classes should match, testing sets are misaligned')
    
    eeg_ppt=ml.drop_ID_cols(eeg_ppt)
    emg_ppt=ml.drop_ID_cols(emg_ppt)
    
    if not args['featfuse_sel_feats_together']:
        '''selecting feats modally before join'''
        eeg_ppt=eeg_ppt.iloc[:,sel_cols_eeg]
        emg_ppt=emg_ppt.iloc[:,sel_cols_emg]
    
    '''joining modalities'''
    eeg_ppt.drop('Label',axis='columns',inplace=True)
    eeg_ppt.rename(columns=lambda x: 'EEG_'+x, inplace=True)
    #emg_others[('EEG_',varname)]=eeg_others[varname] for varname in eeg_others.columns.values()
    labelcol_ppt=emg_ppt.pop('Label')
    emgeeg_ppt=pd.concat([emg_ppt,eeg_ppt],axis=1)
    emgeeg_ppt['Label']=labelcol_ppt
    
    if args['featfuse_sel_feats_together']:
        '''selecting feats after join'''
        emgeeg_ppt=emgeeg_ppt.iloc[:,sel_cols_emgeeg]
        
    predlist_fusion=[]
        
    '''Get values from instances'''
#    IDs=list(emgeeg_ppt.filter(regex='^ID_').keys())
 #   emgeeg_vals=emgeeg_ppt.drop(IDs,axis='columns').values
    
 #   if args['featfuse_sel_feats_together']:
  #     sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_ppt.columns.get_loc('Label'))
   #    emgeeg_ppt = emgeeg_ppt.iloc[:,sel_cols_emgeeg]
    
    emgeeg_vals=emgeeg_ppt.drop(['Label'],axis='columns').values
        
    '''Pass values to models'''    
    distros_fusion=ml.prob_dist(emgeeg_model,emgeeg_vals)
    for distro in distros_fusion:
        pred_fusion=ml.pred_from_distro(classlabels,distro)
        predlist_fusion.append(pred_fusion) 
    
    return targets, predlist_fusion, predlist_fusion, predlist_fusion, classlabels


def fusion_hierarchical(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others, eeg_others = sort_harmonise_data(emg_others,eeg_others)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    
    if args['trialmode']=='WithinPpt':
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_others),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_others).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_others).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    eeg_preds_hierarch= []
    for i, (index_ML, index_Fus) in enumerate(folds.split(eeg_others,eeg_others['ID_splitIndex'])):
        eeg_train_split_ML=eeg_others.iloc[index_ML]
        eeg_train_split_fusion=eeg_others.iloc[index_Fus]
        
        '''Train EEG model'''
        eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
        #sel_cols_eeg=feats.sel_percent_feats_df(eeg_train_split_ML,percent=3)
        eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
        
        eeg_model = ml.train_optimise(eeg_train_split_ML, args['eeg']['eeg_model_type'], args['eeg'])
        classlabels=eeg_model.classes_
        
        '''Get values from instances'''  
        IDs=list(eeg_train_split_fusion.filter(regex='^ID_').keys())
        eeg_train_split_fusion=eeg_train_split_fusion.drop(IDs,axis='columns')
        eeg_train_split_fusion=eeg_train_split_fusion.iloc[:,sel_cols_eeg]
        eegvals=eeg_train_split_fusion.drop(['Label'],axis='columns').values
    
        '''Get EEG preds for EMG training'''
        
        distros_eeg=ml.prob_dist(eeg_model,eegvals)
        if not args['stack_distros']:
            predlist=[]
            for distro in distros_eeg:
                predlist.append(ml.pred_from_distro(classlabels,distro))
            onehot=fusion.setup_onehot(classlabels)
            onehot_pred_eeg=fusion.encode_preds_onehot(eeg_preds_hierarch,onehot)
            preds=ml.pd.DataFrame(onehot_pred_eeg,index=index_Fus,columns=[('EEGOnehotClass'+str(col)) for col in classlabels])
        else:
            preds=ml.pd.DataFrame(distros_eeg,index=index_Fus,columns=[('EEGProbClass'+str(col)) for col in classlabels])
        if len(eeg_preds_hierarch)==0:
            eeg_preds_hierarch=preds
        else:
            eeg_preds_hierarch=ml.pd.concat([eeg_preds_hierarch,preds],axis=0)
    
    eeg_others=ml.drop_ID_cols(eeg_others)
    eeg_others=eeg_others.iloc[:,sel_cols_eeg]
    eeg_model=ml.train_optimise(eeg_others, args['eeg']['eeg_model_type'], args['eeg'])
    classlabels=eeg_model.classes_
    
    emg_train=emg_others
    emg_train=ml.drop_ID_cols(emg_train)
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[emg_train.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    lab=emg_train.pop('Label')
    emg_train=ml.pd.concat([emg_train,eeg_preds_hierarch],axis=1)
    emg_train['Label']=lab  
      
    '''Train EMG model'''
    emg_model=ml.train_optimise(emg_train,args['emg']['emg_model_type'],args['emg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg,eeg = sort_harmonise_data(emg_ppt,eeg_ppt)
    
    predlist_hierarch, predlist_eeg, targets = [], [], []
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')

    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    eeg=eeg.drop(IDs,axis='columns')
    eeg=eeg.iloc[:,sel_cols_eeg]
    eegvals=eeg.drop(['Label'],axis='columns').values    
    
    '''Get EEG Predictions'''
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_eeg.append(pred_eeg)    
    
    emg=emg.drop(IDs,axis='columns') #drop BEFORE inserting EEGOnehot
    emg=emg.iloc[:,sel_cols_emg]
    if not args['stack_distros']:
        '''Add EEG Preds to EMG set'''
        onehot_pred_eeg=fusion.encode_preds_onehot(predlist_eeg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGOnehotClass'+str(lab)),onehot_pred_eeg[:,idx])
            #emg[('EMG1hotClass'+str(lab))]=onehot_pred_eeg[:,idx]
    else:
        '''Add EEG distros to EMG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(emg.columns)
            emg.insert(labelcol-1,('EEGProbClass'+str(lab)),distros_eeg[:,idx])
        
    emg=emg.drop(['Label'],axis='columns')
 
    distros_emg=ml.prob_dist(emg_model,emg.values)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_emg)
    predlist_emg=predlist_hierarch
    
    return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels


def fusion_hierarchical_inv(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others, eeg_others = sort_harmonise_data(emg_others, eeg_others)
            
    emg_others['ID_splitIndex']=emg_others['Label'].astype(str)+emg_others['ID_pptID'].astype(str)
    eeg_others['ID_splitIndex']=eeg_others['Label'].astype(str)+eeg_others['ID_pptID'].astype(str)
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_others),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_others).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[ml.drop_ID_cols(emg_others).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
	
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    emg_preds_hierarch= []
    for i, (index_ML, index_Fus) in enumerate(folds.split(emg_others,emg_others['ID_splitIndex'])):
	
        emg_train_split_ML=emg_others.iloc[index_ML]
        emg_train_split_fusion=emg_others.iloc[index_Fus]
		
        '''Train EMG model'''
        emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
        emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
    
        emg_model = ml.train_optimise(emg_train_split_ML, args['emg']['emg_model_type'], args['emg'])
        classlabels=emg_model.classes_
    
        '''Get values from instances'''
        IDs=list(emg_train_split_fusion.filter(regex='^ID_').keys())
        emg_train_split_fusion=emg_train_split_fusion.drop(IDs,axis='columns')
        emg_train_split_fusion=emg_train_split_fusion.iloc[:,sel_cols_emg]
        emgvals=emg_train_split_fusion.drop(['Label'],axis='columns').values

    
        '''Get EMG preds for EEG training'''
        distros_emg=ml.prob_dist(emg_model,emgvals)
        if not args['stack_distros']:    
            '''Add EMG preds to EEG training set'''
            predlist=[]
            for distro in distros_emg:
                predlist.append(ml.pred_from_distro(classlabels,distro))
            onehot=fusion.setup_onehot(classlabels)
            onehot_pred_emg=fusion.encode_preds_onehot(emg_preds_hierarch,onehot)
            preds=ml.pd.DataFrame(onehot_pred_emg,index=index_Fus,columns=[('EMGOnehotClass'+str(col)) for col in classlabels])
        else:
            '''Add EMG distros to EEG training set'''
            preds=ml.pd.DataFrame(distros_emg,index=index_Fus,columns=[('EMGProbClass'+str(col)) for col in classlabels])
        if len(emg_preds_hierarch)==0:
            emg_preds_hierarch=preds
        else:
            emg_preds_hierarch=ml.pd.concat([emg_preds_hierarch,preds],axis=0)
	
    emg_others=ml.drop_ID_cols(emg_others)
    emg_others=emg_others.iloc[:,sel_cols_emg]
    emg_model=ml.train_optimise(emg_others,args['emg']['emg_model_type'],args['emg'])
    classlabels=emg_model.classes_
	
    eeg_train=eeg_others
    eeg_train=ml.drop_ID_cols(eeg_train)
    if args['trialmode']=='WithinPpt':
        sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[eeg_train.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
	
    lab=eeg_train.pop('Label')
    eeg_train=ml.pd.concat([eeg_train,emg_preds_hierarch],axis=1)
    eeg_train['Label']=lab  
	
    '''Train EEG model'''
    eeg_model=ml.train_optimise(eeg_train,args['eeg']['eeg_model_type'],args['eeg'])
 
    '''-----------------'''
 
    '''TESTING ON PPT DATA'''
    emg, eeg = sort_harmonise_data(emg_ppt, eeg_ppt)
    
    predlist_hierarch, predlist_emg, targets = [], [], []
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')
     
    '''Get values from instances'''
    IDs=list(eeg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    emg=emg.iloc[:,sel_cols_emg]
    emgvals=emg.drop(['Label'],axis='columns').values    
    
   
    '''Get EMG Predictions'''
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    eeg=eeg.drop(IDs,axis='columns') #drop BEFORE inserting EMGOnehot
    eeg=eeg.iloc[:,sel_cols_eeg]
    if not args['stack_distros']:     
        '''Add EMG Preds to EEG set'''
        onehot_pred_emg=fusion.encode_preds_onehot(predlist_emg,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emg[:,idx])
    else:
        '''Add EMG distros to EEG set'''
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg.columns)
            eeg.insert(labelcol-1,('EMGProbClass'+str(lab)),distros_emg[:,idx])
    
    eeg=eeg.drop(['Label'],axis='columns')
 
    distros_eeg=ml.prob_dist(eeg_model,eeg.values)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_hierarch.append(pred_eeg)
    predlist_eeg=predlist_hierarch
    
    if args['get_train_acc']:
        predlist_emgtrain=[]  
        predlist_fustrain=[]        
        eeg_train=eeg_others.drop(IDs,axis='columns')
        eeg_train=eeg_train.iloc[:,sel_cols_eeg]
        emg_train=emg_others.drop(IDs,axis='columns')
        emg_train=emg_train.iloc[:,sel_cols_emg]
        traintargs=eeg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns')#does this need to be .values?
        distros_emgtrain=ml.prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=ml.pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        onehot_pred_emgtrain=fusion.encode_preds_onehot(predlist_emgtrain,onehot)
        for idx,lab in enumerate(classlabels):
            labelcol=len(eeg_train.columns)
            eeg_train.insert(labelcol-1,('EMGOnehotClass'+str(lab)),onehot_pred_emgtrain[:,idx])
        eeg_train=eeg_train.drop(['Label'],axis='columns')
        distros_eegtrain=ml.prob_dist(eeg_model,eeg_train.values)
        for distro in distros_eegtrain:
            pred_eegtrain=ml.pred_from_distro(classlabels,distro)
            predlist_fustrain.append(pred_eegtrain)
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels, traintargs, predlist_fustrain
   
    else:    
        return targets, predlist_emg, predlist_eeg, predlist_hierarch, classlabels
    

def only_EMG(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others, eeg_others = sort_harmonise_data(emg_others, eeg_others)

    '''Select EMG features'''
    emg_train=ml.drop_ID_cols(emg_others)
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
        sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_emg=[emg_train.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    '''Train EMG model'''
    emg_model = ml.train_optimise(emg_train, args['emg']['emg_model_type'], args['emg'])
    classlabels=emg_model.classes_   
 
 
    '''TESTING ON PPT DATA'''    
    emg, eeg = sort_harmonise_data(emg_ppt,eeg_ppt)
    
    predlist_emg, targets = [], []
    
    if emg['Label'].equals(eeg['Label']):
        targets=emg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')

    '''Get values from instances'''
    IDs=list(eeg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    emg=emg.iloc[:,sel_cols_emg]
    emgvals=emg.drop(['Label'],axis='columns').values    
    
    '''Get EMG Predictions'''
    distros_emg=ml.prob_dist(emg_model,emgvals)
    for distro in distros_emg:
        pred_emg=ml.pred_from_distro(classlabels,distro)
        predlist_emg.append(pred_emg)
    
    if args['get_train_acc']:
        predlist_emgtrain=[]
        traintargs=emg_train['Label'].values.tolist()
        emgtrainvals=emg_train.drop('Label',axis='columns') #why DOESNT this need to be .values?
        distros_emgtrain=ml.prob_dist(emg_model,emgtrainvals)
        for distro in distros_emgtrain:
            pred_emgtrain=ml.pred_from_distro(classlabels,distro)
            predlist_emgtrain.append(pred_emgtrain)
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels, traintargs, predlist_emgtrain
   
    else:
        return targets, predlist_emg, predlist_emg, predlist_emg, classlabels


def only_EEG(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    '''TRAINING ON NON-PPT DATA'''
    emg_others, eeg_others = sort_harmonise_data(emg_others, eeg_others)

    '''Select EEG features'''
    eeg_train=ml.drop_ID_cols(eeg_others)
    if args['trialmode']=='WithinPpt':
        sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[eeg_train.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
    
    '''Train EEG model'''
    eeg_model = ml.train_optimise(eeg_train, args['eeg']['eeg_model_type'], args['eeg'])
    classlabels=eeg_model.classes_   
 
 
    '''TESTING ON PPT DATA'''    
    emg, eeg = sort_harmonise_data(emg_ppt,eeg_ppt)
    
    predlist_eeg, targets = [], []
    
    if eeg['Label'].equals(emg['Label']):
        targets=eeg['Label'].values.tolist()
    else:
        raise Exception('Sense check failed, target label should agree between modes')        
     
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    eeg=eeg.drop(IDs,axis='columns')
    eeg=eeg.iloc[:,sel_cols_eeg]
    eegvals=eeg.drop(['Label'],axis='columns').values    
    
    '''Get EEG Predictions'''
    distros_eeg=ml.prob_dist(eeg_model,eegvals)
    for distro in distros_eeg:
        pred_eeg=ml.pred_from_distro(classlabels,distro)
        predlist_eeg.append(pred_eeg)
        
    if args['get_train_acc']:
        predlist_eegtrain=[]
        traintargs=eeg_train['Label'].values.tolist()
        eegtrainvals=eeg_train.drop('Label',axis='columns') #why DOESNT this need to be .values?
        distros_eegtrain=ml.prob_dist(eeg_model,eegtrainvals)
        for distro in distros_eegtrain:
            pred_eegtrain=ml.pred_from_distro(classlabels,distro)
            predlist_eegtrain.append(pred_eegtrain)
        return targets, predlist_eeg, predlist_eeg, predlist_eeg, classlabels, traintargs, predlist_eegtrain
   
    else:
        return targets, predlist_eeg, predlist_eeg, predlist_eeg, classlabels


def fusion_metamodel(emg_train, eeg_train, emg_test, eeg_test, args):
        
    emg_train, eeg_train = sort_harmonise_data(emg_train,eeg_train)
            
    emg_train['ID_splitIndex']=emg_train['Label'].astype(str)+emg_train['ID_pptID'].astype(str)
    eeg_train['ID_splitIndex']=eeg_train['Label'].astype(str)+eeg_train['ID_pptID'].astype(str)
    
    if args['trialmode']=='WithinPpt':
        sel_cols_emg=feats.sel_percent_feats_df(ml.drop_ID_cols(emg_train),percent=15)
        sel_cols_emg=np.append(sel_cols_emg,ml.drop_ID_cols(emg_train).columns.get_loc('Label'))
        sel_cols_eeg=feats.sel_feats_l1_df(ml.drop_ID_cols(eeg_train),sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
        sel_cols_eeg=np.append(sel_cols_eeg,ml.drop_ID_cols(eeg_train).columns.get_loc('Label'))
    elif args['trialmode']=='LOO':
        idx=int(eeg_test['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[ml.drop_ID_cols(eeg_train).columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[idx].tolist()]
        sel_cols_emg=[ml.drop_ID_cols(emg_train).columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[idx].tolist()]
    
    random_split=random.randint(0,100)
    folds=StratifiedKFold(random_state=random_split,n_splits=3,shuffle=False)
    fustargets=[]
    fusdistros_emg=[]
    fusdistros_eeg=[]
    for i, (index_ML, index_Fus) in enumerate(folds.split(emg_train,emg_train['ID_splitIndex'])):
        emg_train_split_ML=emg_train.iloc[index_ML]
        emg_train_split_fusion=emg_train.iloc[index_Fus]
        eeg_train_split_ML=eeg_train.iloc[index_ML]
        eeg_train_split_fusion=eeg_train.iloc[index_Fus]
        
        emg_train_split_ML=ml.drop_ID_cols(emg_train_split_ML)
        eeg_train_split_ML=ml.drop_ID_cols(eeg_train_split_ML)
        
        emg_train_split_ML=emg_train_split_ML.iloc[:,sel_cols_emg]
        eeg_train_split_ML=eeg_train_split_ML.iloc[:,sel_cols_eeg]
        
        emg_model,eeg_model=train_models_opt(emg_train_split_ML,eeg_train_split_ML,args)    
        classlabels = emg_model.classes_
        
        targets,predlist_emg,predlist_eeg,_,distros_emg,distros_eeg,_=refactor_synced_predict(emg_train_split_fusion, eeg_train_split_fusion, emg_model, eeg_model, classlabels, args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if len(fustargets)==0:
            fustargets=targets
            fusdistros_emg=distros_emg
            fusdistros_eeg=distros_eeg
        else:
            fustargets=fustargets+targets
            fusdistros_emg=np.concatenate((fusdistros_emg,distros_emg),axis=0)
            fusdistros_eeg=np.concatenate((fusdistros_eeg,distros_eeg),axis=0)
            
    emg_train=ml.drop_ID_cols(emg_train)
    emg_train=emg_train.iloc[:,sel_cols_emg]
    eeg_train=ml.drop_ID_cols(eeg_train)
    eeg_train=eeg_train.iloc[:,sel_cols_eeg]
    emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
    
    if not args['stack_distros']:
        onehotEncoder=fusion.setup_onehot(classlabels)
        onehot_pred_emg=fusion.encode_preds_onehot(fusdistros_emg,onehotEncoder)
        onehot_pred_eeg=fusion.encode_preds_onehot(fusdistros_eeg,onehotEncoder)
        if args['fusion_alg']=='svm':
            fuser=fusion.train_svm_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['svmfuse'])
        elif args['fusion_alg']=='lda':
            fuser=fusion.train_lda_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['ldafuse'])
        elif args['fusion_alg']=='rf':
            fuser=fusion.train_rf_fuser(onehot_pred_emg, onehot_pred_eeg, fustargets, args['RFfuse'])
    else:
        onehotEncoder=None
        if args['fusion_alg']=='svm':
            fuser=fusion.train_svm_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['svmfuse'])
        elif args['fusion_alg']=='lda':
            fuser=fusion.train_lda_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['ldafuse'])
        elif args['fusion_alg']=='rf':
            fuser=fusion.train_rf_fuser(fusdistros_emg, fusdistros_eeg, fustargets, args['RFfuse'])

    
    '---------------'
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
    targets, predlist_emg, predlist_eeg, _, distros_emg, distros_eeg, _  = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
    
    if args['stack_distros']:
        if args['fusion_alg']=='svm':
            predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
        elif args['fusion_alg']=='lda':
            predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
        elif args['fusion_alg']=='rf':
            predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,distros_emg,distros_eeg,classlabels)
    else:
        if args['fusion_alg']=='svm':
            predlist_fusion=fusion.svm_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
        elif args['fusion_alg']=='lda':
            predlist_fusion=fusion.lda_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
        elif args['fusion_alg']=='rf':
            predlist_fusion=fusion.rf_fusion(fuser,onehotEncoder,predlist_emg,predlist_eeg,classlabels)
    
    if args['get_train_acc']:
        emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)                
        traintargs, predlist_emgtrain, predlist_eegtrain, _, distros_emgtrain, distros_eegtrain, _ = refactor_synced_predict(emg_train, eeg_train, emg_model, eeg_model, classlabels,args,sel_cols_eeg,sel_cols_emg,get_distros=args['stack_distros'])
        if args['stack_distros']:
            if args['fusion_alg']=='svm':
                predlist_train=fusion.svm_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
            elif args['fusion_alg']=='lda':
                predlist_train=fusion.lda_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)
            elif args['fusion_alg']=='rf':
                predlist_train=fusion.rf_fusion(fuser,onehotEncoder,distros_emgtrain,distros_eegtrain,classlabels)    
        else:
            if args['fusion_alg']=='svm':
                predlist_train=fusion.svm_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
            elif args['fusion_alg']=='lda':
                predlist_train=fusion.lda_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
            elif args['fusion_alg']=='rf':
                predlist_train=fusion.rf_fusion(fuser,onehotEncoder,predlist_emgtrain,predlist_eegtrain,classlabels)
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train  
    else:
        return targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels



def get_joint_feats(emg_others,eeg_others,args):
    emg_others, eeg_others = sort_harmonise_data(emg_others, eeg_others)
    
    if emg_others['Label'].equals(eeg_others['Label']):
        pass
    else:
        raise RuntimeError('Target classes should match, training sets are misaligned')
    
    eeg_others=ml.drop_ID_cols(eeg_others)
    emg_others=ml.drop_ID_cols(emg_others)

    eeg_others.drop('Label',axis='columns',inplace=True)
    eeg_others.rename(columns=lambda x: 'EEG_'+x, inplace=True)
    labelcol=emg_others.pop('Label')
    emgeeg_others=pd.concat([emg_others,eeg_others],axis=1)
    emgeeg_others['Label']=labelcol
    
    sel_cols_emgeeg=feats.sel_feats_l1_df(emgeeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats']+88)
    '''here we are taking total features = N(EEG feats) + N(EMG feats) = N(EEG) + 88'''
    sel_cols_emgeeg=np.append(sel_cols_emgeeg,emgeeg_others.columns.get_loc('Label'))

    emgeeg_others = emgeeg_others.iloc[:,sel_cols_emgeeg]
       
    return sel_cols_emgeeg, emgeeg_others.columns.values



def get_LOO_feats(args,jointly=False):

    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
        
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    

    if jointly==True:
        joint_feat_idxs, joint_feat_names = [], []
        
        for idx,emg_mask in enumerate(emg_masks):
            eeg_mask=eeg_masks[idx]
            
            emg_others = emg_set[~emg_mask]
            eeg_others = eeg_set[~eeg_mask]
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
            
            colsidx,colnames=get_joint_feats(emg_others, eeg_others, args)
            joint_feat_idxs.append(colsidx)
            joint_feat_names.append(colnames)
        
        joint_feats_idx_df=pd.DataFrame(joint_feat_idxs)
        joint_feats_idx_df.to_csv(params.joint_LOO_feat_idx_csv,index=False,header=False)
        joint_feats_df=pd.DataFrame(joint_feat_names)
        joint_feats_df.to_csv(params.joint_LOO_feats_csv,index=False,header=False)
        
    elif jointly==False:
        eeg_feat_idxs, eeg_feat_names = [], []
        emg_feat_idxs, emg_feat_names = [], [] 
        
        for idx,emg_mask in enumerate(emg_masks):
            eeg_mask=eeg_masks[idx]
            
            emg_others = emg_set[~emg_mask]
            eeg_others = eeg_set[~eeg_mask]
            
            if args['scalingtype']:
                emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
                eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
            
                            
            emg_others=ml.drop_ID_cols(emg_others)
            eeg_others=ml.drop_ID_cols(eeg_others)
            
            sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
            
            sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))
                
            eeg_others=eeg_others.iloc[:,sel_cols_eeg]
            emg_others=emg_others.iloc[:,sel_cols_emg]

            eeg_feat_idxs.append(sel_cols_eeg)
            eeg_feat_names.append(eeg_others.columns.values)
            emg_feat_idxs.append(sel_cols_emg)
            emg_feat_names.append(emg_others.columns.values)

        eeg_feats_idx_df=pd.DataFrame(eeg_feat_idxs)
        eeg_feats_idx_df.to_csv(params.eeg_LOO_feat_idx_csv,index=False,header=False)
        eeg_feats_df=pd.DataFrame(eeg_feat_names)
        eeg_feats_df.to_csv(params.eeg_LOO_feats_csv,index=False,header=False)
        
        emg_feats_idx_df=pd.DataFrame(emg_feat_idxs)
        emg_feats_idx_df.to_csv(params.emg_LOO_feat_idx_csv,index=False,header=False)
        emg_feats_df=pd.DataFrame(emg_feat_names)
        emg_feats_df.to_csv(params.emg_LOO_feats_csv,index=False,header=False)




def function_fuse_LOO(args):
    start=time.time()

    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
        
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs, emg_accs, eeg_accs, train_accs = [], [], [], []
    
    kappas=[]
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        emg_others = emg_set[~emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        eeg_others = eeg_set[~eeg_mask]
        
        if args['scalingtype']:
            emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
            eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
            emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
            eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
        
        if args['fusion_alg'] in ['svm','lda','rf']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_metamodel(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
        elif args['fusion_alg']=='hierarchical':            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_others, eeg_others, emg_ppt, eeg_ppt, args)

        elif args['fusion_alg']=='hierarchical_inv':          
                                                  
            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                 
        elif args['fusion_alg']=='featlevel':  
                            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=feature_fusion(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
        elif args['fusion_alg']=='just_emg':
            
            if not args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        
        elif args['fusion_alg']=='just_eeg':
            
            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                    
        else:
                
            emg_others=ml.drop_ID_cols(emg_others)
            eeg_others=ml.drop_ID_cols(eeg_others)
            
            if args['trialmode']=='WithinPpt':
                sel_cols_eeg=feats.sel_feats_l1_df(eeg_others,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
                sel_cols_eeg=np.append(sel_cols_eeg,eeg_others.columns.get_loc('Label'))
                
                sel_cols_emg=feats.sel_percent_feats_df(emg_others,percent=15)
                sel_cols_emg=np.append(sel_cols_emg,emg_others.columns.get_loc('Label'))
            elif args['trialmode']=='LOO':
                pptidx=int(emg_ppt['ID_pptID'].iloc[0])-1
                sel_cols_eeg=[eeg_others.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[pptidx].tolist()]
                sel_cols_emg=[emg_others.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[pptidx].tolist()]
                
            eeg_others=eeg_others.iloc[:,sel_cols_eeg]
            emg_others=emg_others.iloc[:,sel_cols_emg]
            
            emg_model,eeg_model=train_models_opt(emg_others,eeg_others,args)
        
            classlabels = emg_model.classes_
            
            emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_= refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)
        
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
                
        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
            tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_fusion))
        
        if args['get_train_acc']:
            train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
            train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
    
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)
    mean_eeg=stats.mean(eeg_accs)
    median_eeg=stats.median(eeg_accs)
    median_kappa=stats.median(kappas)
    mean_train_acc=stats.mean(train_accs)
    end=time.time()
    return {
        'loss': 1-mean_acc,
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'fusion_mean_acc':mean_acc,
        'fusion_median_acc':median_acc,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'eeg_mean_acc':mean_eeg,
        'eeg_median_acc':median_eeg,
        'emg_accs':emg_accs,
        'eeg_accs':eeg_accs,
        'fusion_accs':accs,
        'mean_train_acc':mean_train_acc,
        'elapsed_time':end-start,}

def function_fuse_withinppt(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        eeg_set_path=args['eeg_set_path']
        emg_set=ml.pd.read_csv(emg_set_path,delimiter=',')
        eeg_set=ml.pd.read_csv(eeg_set_path,delimiter=',')
    else:
        emg_set=args['emg_set']
        eeg_set=args['eeg_set']
    if not args['prebalanced']: 
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
    
    eeg_masks=get_ppt_split(eeg_set,args)
    emg_masks=get_ppt_split(emg_set,args)
    
    accs, emg_accs, eeg_accs, train_accs = [], [], [], []
   
    kappas=[]
    for idx,emg_mask in enumerate(emg_masks):
        eeg_mask=eeg_masks[idx]
        
        emg_ppt = emg_set[emg_mask]
        eeg_ppt = eeg_set[eeg_mask]
        
        emg_ppt,eeg_ppt = sort_harmonise_data(emg_ppt,eeg_ppt)
        
        if not emg_ppt['ID_stratID'].equals(eeg_ppt['ID_stratID']):
            raise ValueError('EMG & EEG performances misaligned')
        
        eeg_ppt['ID_stratID']=eeg_ppt['ID_run'].astype(str)+eeg_ppt['Label'].astype(str)+eeg_ppt['ID_gestrep'].astype(str)
        emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_gestrep'].astype(str)

        random_split=random.randint(0,100)
        
        gest_perfs=emg_ppt['ID_stratID'].unique()
        gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
        train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])
        
        eeg_train=eeg_ppt[eeg_ppt['ID_stratID'].isin(train_split[0])]
        eeg_test=eeg_ppt[eeg_ppt['ID_stratID'].isin(test_split[0])]
        emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
        emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]

        
        if args['scalingtype']:
            emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
            eeg_train,eegscaler=feats.scale_feats_train(eeg_train,args['scalingtype'])
            emg_test=feats.scale_feats_test(emg_test,emgscaler)
            eeg_test=feats.scale_feats_test(eeg_test,eegscaler)

        if args['fusion_alg'] in ['svm','lda','rf']:
            if args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_metamodel(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_metamodel(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='hierarchical':                            
                        
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical(emg_train, eeg_train, emg_test, eeg_test, args)

        elif args['fusion_alg']=='hierarchical_inv':

            if not args['get_train_acc']:            
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train = fusion_hierarchical_inv(emg_train, eeg_train, emg_test, eeg_test, args)
                 
        elif args['fusion_alg']=='featlevel':
                            
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=feature_fusion(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='just_emg':
            
            if not args['get_train_acc']:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EMG(emg_train, eeg_train, emg_test, eeg_test, args)
        
        elif args['fusion_alg']=='just_eeg':

            if not args['get_train_acc']:    
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
            else:
                targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=only_EEG(emg_train, eeg_train, emg_test, eeg_test, args)
        else:
            
            if args['get_train_acc']:
                emg_trainacc=emg_train.copy()
                eeg_trainacc=eeg_train.copy()
                emg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                eeg_trainacc.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
           
            
            emg_train=ml.drop_ID_cols(emg_train)
            eeg_train=ml.drop_ID_cols(eeg_train)
            
            sel_cols_eeg=feats.sel_feats_l1_df(eeg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
            sel_cols_eeg=np.append(sel_cols_eeg,eeg_train.columns.get_loc('Label'))
            eeg_train=eeg_train.iloc[:,sel_cols_eeg]
            
            sel_cols_emg=feats.sel_percent_feats_df(emg_train,percent=15)
            sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
            emg_train=emg_train.iloc[:,sel_cols_emg]
            
            emg_model,eeg_model=train_models_opt(emg_train,eeg_train,args)
        
            classlabels = emg_model.classes_
            
            emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            eeg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
                
            targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_ = refactor_synced_predict(emg_test, eeg_test, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

            if args['get_train_acc']:
                traintargs, predlist_emgtrain, predlist_eegtrain, predlist_train,_,_,_ = refactor_synced_predict(emg_trainacc, eeg_trainacc, emg_model, eeg_model, classlabels, args, sel_cols_eeg,sel_cols_emg)
        
        gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
                
        if args['plot_confmats']:
            gesturelabels=[params.idx_to_gestures[label] for label in classlabels]
            tt.confmat(gest_truth,gest_pred_eeg,gesturelabels,title='EEG')
            tt.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
            tt.confmat(gest_truth,gest_pred_fusion,gesturelabels,title='Fusion')
            
        emg_accs.append(accuracy_score(gest_truth,gest_pred_emg))
        eeg_accs.append(accuracy_score(gest_truth,gest_pred_eeg))
        accs.append(accuracy_score(gest_truth,gest_pred_fusion))
        
        kappas.append(cohen_kappa_score(gest_truth,gest_pred_fusion))
        
        if args['get_train_acc']:
            train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
            train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
            train_accs.append(accuracy_score(train_truth,train_preds))
        else:
            train_accs.append(0)
        
    mean_acc=stats.mean(accs)
    median_acc=stats.median(accs)
    mean_emg=stats.mean(emg_accs)
    median_emg=stats.median(emg_accs)
    mean_eeg=stats.mean(eeg_accs)
    median_eeg=stats.median(eeg_accs)
    median_kappa=stats.median(kappas)
    mean_train_acc=stats.mean(train_accs)
    end=time.time()
    return {
        'loss': 1-mean_acc,
        'status': STATUS_OK,
        'median_kappa':median_kappa,
        'fusion_mean_acc':mean_acc,
        'fusion_median_acc':median_acc,
        'emg_mean_acc':mean_emg,
        'emg_median_acc':median_emg,
        'eeg_mean_acc':mean_eeg,
        'eeg_median_acc':median_eeg,
        'emg_accs':emg_accs,
        'eeg_accs':eeg_accs,
        'fusion_accs':accs,
        'mean_train_acc':mean_train_acc,
        'elapsed_time':end-start,}

        
def setup_search_space(architecture,include_svm):
    emgoptions=[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.RF.ntrees',10,100,q=5)),
                 'max_depth':5,
                 },
                {'emg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('emg.knn.k',1,25,q=1)),
                 },
                {'emg_model_type':'LDA',
                 'LDA_solver':hp.choice('emg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('emg.lda.shrinkage',0.0,1.0),
                 },
                {'emg_model_type':'QDA',
                 'regularisation':hp.uniform('emg.qda.regularisation',0.0,1.0),
                 },
                {'emg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('emg.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                 },
                ]
    eegoptions=[
                {'eeg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('eeg_ntrees',10,100,q=5)),
                 'max_depth':5,
                 },
                {'eeg_model_type':'kNN',
                 'knn_k':scope.int(hp.quniform('eeg.knn.k',1,25,q=1)),
                 },
                {'eeg_model_type':'LDA',
                 'LDA_solver':hp.choice('eeg.LDA_solver',['svd','lsqr','eigen']),
                 'shrinkage':hp.uniform('eeg.lda.shrinkage',0.0,1.0),
                 },
                {'eeg_model_type':'QDA',
                 'regularisation':hp.uniform('eeg.qda.regularisation',0.0,1.0),
                 },
                {'eeg_model_type':'gaussNB',
                 'smoothing':hp.loguniform('eeg.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                 },
                ]
    if include_svm:
        emgoptions.append({'emg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('emg.svm.kernel',['rbf']),
                 'svm_C':hp.loguniform('emg.svm.c',np.log(0.1),np.log(100)),
                 'gamma':hp.loguniform('emg.svm.gamma',np.log(0.01),np.log(1)),
                 })
        eegoptions.append({'eeg_model_type':'SVM_PlattScale',
                 'kernel':hp.choice('eeg.svm.kernel',['rbf']),
                 'svm_C':hp.loguniform('eeg.svm.c',np.log(0.1),np.log(100)),
                 'gamma':hp.loguniform('eeg.svm.gamma',np.log(0.01),np.log(1)), 
                 })
    
    space = {
            #modelling options
            'emg':hp.choice('emg model',emgoptions),
            'eeg':hp.choice('eeg model',eegoptions),
            'svmfuse':{
                'svm_C':hp.loguniform('fus.svm.c',np.log(0.01),np.log(100)),
                },
            'ldafuse':{
                'LDA_solver':hp.choice('fus.LDA.solver',['svd','lsqr','eigen']),
                'shrinkage':hp.uniform('fus.lda.shrinkage',0.0,1.0),
                },
            'RFfuse':{
                'n_trees':scope.int(hp.quniform('fus.RF.ntrees',10,100,q=5)),
                'max_depth':5,
                },
            'eeg_weight_opt':hp.uniform('fus.optEEG.EEGweight',0.0,100.0),
            'fusion_alg':hp.choice('fusion algorithm',[
                'mean',
                '3_1_emg',
                '3_1_eeg',
                'opt_weight',
                'highest_conf',
                'svm',
                'lda',
                'rf',
                ]),
            #dataset paths
            'emg_set_path':params.emg_no_holdout,
            'eeg_set_path':params.eeg_no_holdout,
            # variables to determine experimental steps
            'using_literature_data':True,
            'data_in_memory':False, # have the datasets been loaded into the dict yet
            'prebalanced':False, # are the datasets balanced already
            'scalingtype':'standardise', #'normalise' and None also valid options
            'plot_confmats':False, 
            'get_train_acc':False,
            'stack_distros':True,#perform any stacking with probabilities, not predictions
            }
    
    if architecture=='featlevel':
        if include_svm:
            space.update({
                'fusion_alg':hp.choice('fusion algorithm',['featlevel',]),
                'featfuse_sel_feats_together':True, #this needs to be updated MANUALLY for joint vs sep sel
                'featfuse':hp.choice('featfuse model',[
                    {'featfuse_model_type':'RF',
                     'n_trees':scope.int(hp.quniform('featfuse.RF.ntrees',10,100,q=5)),
                     'max_depth':5,
                     },
                    {'featfuse_model_type':'kNN',
                     'knn_k':scope.int(hp.quniform('featfuse.knn.k',1,25,q=1)),
                     },
                    {'featfuse_model_type':'LDA',
                     'LDA_solver':hp.choice('featfuse.LDA_solver',['svd','lsqr','eigen']),
                     'shrinkage':hp.uniform('featfuse.lda.shrinkage',0.0,1.0),
                     },
                    {'featfuse_model_type':'QDA',
                     'regularisation':hp.uniform('featfuse.qda.regularisation',0.0,1.0),
                     },
                    {'featfuse_model_type':'gaussNB',
                     'smoothing':hp.loguniform('featfuse.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                     },
                    {'featfuse_model_type':'SVM_PlattScale',
                     'kernel':hp.choice('featfuse.svm.kernel',['rbf']),
                     'svm_C':hp.loguniform('featfuse.svm.c',np.log(0.1),np.log(100)),
                     'gamma':hp.loguniform('featfuse.svm.gamma',np.log(0.01),np.log(1)),
                     },
                    ]),
                })
        else:
            space.update({
                'fusion_alg':hp.choice('fusion algorithm',['featlevel',]),
                'featfuse_sel_feats_together':True, #this needs to be updated MANUALLY for joint vs sep sel
                'featfuse':hp.choice('featfuse model',[
                    {'featfuse_model_type':'RF',
                     'n_trees':scope.int(hp.quniform('featfuse.RF.ntrees',10,100,q=5)),
                     'max_depth':5,
                     },
                    {'featfuse_model_type':'kNN',
                     'knn_k':scope.int(hp.quniform('featfuse.knn.k',1,25,q=1)),
                     },
                    {'featfuse_model_type':'LDA',
                     'LDA_solver':hp.choice('featfuse.LDA_solver',['svd','lsqr','eigen']),
                     'shrinkage':hp.uniform('featfuse.lda.shrinkage',0.0,1.0),
                     },
                    {'featfuse_model_type':'QDA',
                     'regularisation':hp.uniform('featfuse.qda.regularisation',0.0,1.0),
                     },
                    {'featfuse_model_type':'gaussNB',
                     'smoothing':hp.loguniform('featfuse.gnb.smoothing',np.log(1e-9),np.log(1e0)),
                     },
                    ]),
                })
        space.pop('emg',None)
        space.pop('eeg',None)
        
    elif architecture=='hierarchical':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['hierarchical',])})
        
    elif architecture=='hierarchical_inv':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['hierarchical_inv',])})
        
    elif architecture=='just_emg':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['just_emg',])})
        space.pop('eeg',None)
    
    elif architecture=='just_eeg':
        space.update({'fusion_alg':hp.choice('fusion algorithm',['just_eeg',])})
        space.pop('emg',None)
        
    if architecture in ['featlevel','hierarchical','hierarchical_inv',
                        'just_emg','just_eeg']:
            space.pop('svmfuse',None)
            space.pop('ldafuse',None)
            space.pop('RFfuse',None)
            space.pop('eeg_weight_opt',None)
        
    return space


def optimise_fusion(trialmode,prebalance=True,architecture='decision',platform='not server',iters=100):
    incl_svm = True if trialmode=='WithinPpt' else False #SVMs not viable for Generalist!
    space=setup_search_space(architecture,incl_svm)
    space.update({'trialmode':trialmode})
    
    if platform=='server':
        #overwriting with appropriate filepath if we're running experiments on the ML server
        space.update({'emg_set_path':params.jeong_EMGnoHO_server,
                      'eeg_set_path':params.jeong_EEGnoHO_server})
    
    if prebalance:
        # we can load and balance datasets here, rather than repeatedly during modelling
        emg_set=ml.pd.read_csv(space['emg_set_path'],delimiter=',')
        eeg_set=ml.pd.read_csv(space['eeg_set_path'],delimiter=',')
        emg_set,eeg_set=balance_set(emg_set,eeg_set)
        space.update({'emg_set':emg_set,'eeg_set':eeg_set,'data_in_memory':True,'prebalanced':True})
        
    trials=Trials()
    
    if trialmode=='LOO':
        space.update({'l1_sparsity':0.005})
        space.update({'l1_maxfeats':88})
        ''' the above not actually used on the fly, only once to determine feats '''
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        emgeegcols=pd.read_csv(params.jointemgeegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,
                      'jointemgeeg_feats_LOO':emgeegcols,})
        
        best = fmin(function_fuse_LOO,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials=trials)      
        
    elif trialmode=='WithinPpt':
        space.update({'l1_sparsity':0.005})
        space.update({'l1_maxfeats':40})
        
        best = fmin(function_fuse_withinppt,
                space=space,
                algo=tpe.suggest,
                max_evals=iters,
                trials=trials)
    else:
        raise ValueError('Unrecognised testing strategy, should be LOO or WithinPpt')
        
    return best, space, trials
    

def save_resultdict(filepath,resultdict,dp=4):
    #drop some stuff we don't want to clog up a results file
    status=resultdict['Results'].pop('status')
    emg_accs=resultdict['Results'].pop('emg_accs',None)
    eeg_accs=resultdict['Results'].pop('eeg_accs',None)
    fusion_accs=resultdict['Results'].pop('fusion_accs',None)
    
    f=open(filepath,'w')
    try:
        target=list(resultdict['Results'].keys())[list(resultdict['Results'].values()).index(1-resultdict['Results']['loss'])]
        f.write(f"Optimising for {target}\n\n")
    except ValueError:
        target, _ = min(resultdict['Results'].items(), key=lambda x: abs(1-resultdict['Results']['loss'] - x[1]))
        f.write(f"Probably optimising for {target}\n\n")
    
    if 'eeg' in resultdict['Chosen parameters']:
        f.write('EEG Parameters:\n')
        for k in resultdict['Chosen parameters']['eeg'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['eeg'][k],dp)if not isinstance(resultdict['Chosen parameters']['eeg'][k],str) else resultdict['Chosen parameters']['eeg'][k]}'\n")
    
    if 'emg' in resultdict['Chosen parameters']:
        f.write('EMG Parameters:\n')
        for k in resultdict['Chosen parameters']['emg'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['emg'][k],dp)if not isinstance(resultdict['Chosen parameters']['emg'][k],str) else resultdict['Chosen parameters']['emg'][k]}'\n")
    
    f.write('Fusion algorithm:\n')
    f.write(f"\t'{'fusion_alg'}':'{resultdict['Chosen parameters']['fusion_alg']}'\n")
    if resultdict['Chosen parameters']['fusion_alg']=='featlevel':
        f.write('Feature-level Fusion Parameters:\n')
        for k in resultdict['Chosen parameters']['featfuse'].keys():
            f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['featfuse'][k],dp)if not isinstance(resultdict['Chosen parameters']['featfuse'][k],str) else resultdict['Chosen parameters']['featfuse'][k]}'\n")
    
    f.write('Results:\n')
    resultdict['Results']['status']=status
    for k in resultdict['Results'].keys():
        f.write(f"\t'{k}':'{round(resultdict['Results'][k],dp)if not isinstance(resultdict['Results'][k],str) else resultdict['Results'][k]}'\n")
    
    #bring back the stuff we dropped earlier!
    resultdict['Results']['emg_accs']=emg_accs
    resultdict['Results']['eeg_accs']=eeg_accs
    resultdict['Results']['fusion_accs']=fusion_accs
    f.close()


def load_results_obj(path):
    load_trials=pickle.load(open(path,'rb'))
    load_table=pd.DataFrame(load_trials.trials)
    load_table_readable=pd.concat(
        [pd.DataFrame(load_table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(load_table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    #we often only want to work with table_readable, so just call as _,_,X=load_results_obj()
    return load_trials,load_table,load_table_readable


if __name__ == '__main__':

    ''' Pass the following arguments in when calling the program:
        architecture
            Architecture for which to perform CASH optimisation.
            Following are supported:
            [decision,featlevel,hierarchical,hierarchical_inv,
             just_emg,just_eeg]
        trialmode
            [WithinPpt]: use a subject-specific Bespoke setting
            [LOO]: use a subject-independent (Leave-One-Out) Generalist setting
        platform
            Specific to our own deployment of experiments.
            Mainly determines which filepaths are pulled from params.py
            [server] means code is running on our dedicated ML server
            [not server] means it isn't, which is probably what you want!
            At the terminal you may need to write this as [not_server]
        num_iters
            How many optimisation iterations to give the CASH procedure.
            All our experiments used 100.
        showplots
            Whether or not to display plots of results or just save them.
            [True] or [False]        
    '''
    if len(sys.argv)>1:
        architecture=sys.argv[1]
        trialmode=sys.argv[2]
        platform=sys.argv[3]
        if platform=='not_server':
            platform='not server'
        if len(sys.argv)>4:
            num_iters=int(sys.argv[4])
        if len(sys.argv)>5:
            showplots=sys.argv[5].lower()
        else:
            showplots=None
        do_NOT_save = False
    else:
        ''' Alternatively, if e.g. running from an IDE, set arguments below'''
        architecture='just_eeg'    
        trialmode='WithinPpt'
        platform='not server'
        num_iters=100
        showplots=None
        do_NOT_save = False
        #switch this to TRUE if you just need to test some functionality without overwriting results
        
    if architecture not in ['decision','featlevel','hierarchical','hierarchical_inv','just_emg','just_eeg']:
        errstring=('requested architecture '+architecture+' not recognised, expecting one of:\n decision\n featlevel\n hierarchical\n hierarchical_inv')
        raise KeyboardInterrupt(errstring)
        
    if (platform=='server') or (showplots=='false'):
        showplot_toggle=False
    else:
        showplot_toggle=True


    best,space,trials=optimise_fusion(trialmode=trialmode,architecture=architecture,platform=platform,iters=num_iters)
    
    
    '''performing a fresh evaluation with the chosen params to get conf mats'''    
    chosen_space=space_eval(space,best)
    chosen_space['plot_confmats']=True
    if trialmode=='LOO':
        chosen_results=function_fuse_LOO(chosen_space)
    elif trialmode=='WithinPpt':
        chosen_results=function_fuse_withinppt(chosen_space)
    
    
    
    best_results=trials.best_trial['result']
    
    bestparams=space_eval(space,best)
    
    for static in ['eeg_set_path','emg_set_path','using_literature_data']:
        bestparams.pop(static)
    bestparams.pop('eeg_set')
    bestparams.pop('emg_set')
    bestparams.pop('eeg_feats_LOO',None)
    bestparams.pop('emg_feats_LOO',None)
    bestparams.pop('jointemgeeg_feats_LOO',None)
    
    print(bestparams)

    print('Best mean Fusion accuracy: ',1-best_results['loss'])
         
    winner={'Chosen parameters':bestparams,
            'Results':best_results}
    
    table=pd.DataFrame(trials.trials)
    table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    
    if do_NOT_save:
        raise
    
    '''SETTING RESULT PATH'''
    currentpath=os.path.dirname(__file__)
    result_dir=params.jeong_results_dir
    resultpath=os.path.join(currentpath,result_dir)    
    resultpath=os.path.join(resultpath,'CASH',trialmode,architecture)
    
    
    '''PICKLING THE TRIALS OBJECT'''
    trials_obj_path=os.path.join(resultpath,'trials_obj.p')
    pickle.dump(trials,open(trials_obj_path,'wb'))
    
    
    '''saving best parameters & results'''
    reportpath=os.path.join(resultpath,'params_results_report.txt')
    save_resultdict(reportpath,winner)
    
    
    if architecture=='featlevel':
        fus_acc_plot=plot_stat_in_time(trials,'fusion_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['fusion_mean_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)  
        fus_acc_box=scatterbox(trials,'fusion_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        fus_acc_plot.savefig(os.path.join(resultpath,'fusion_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'acc_compare.png'))
        fus_acc_box.savefig(os.path.join(resultpath,'fusion_box.png'))
        
        per_fusalg=boxplot_param(table_readable,'featfuse model','fusion_mean_acc',showplot=showplot_toggle)
        per_fusalg.savefig(os.path.join(resultpath,'fus_alg.png'))
        
    elif architecture=='just_emg':
        emg_acc_plot=plot_stat_in_time(trials,'emg_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc'],runbest='emg_mean_acc',showplot=showplot_toggle)  
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','mean_train_acc'],runbest='emg_mean_acc',showplot=showplot_toggle)
        emg_acc_box=scatterbox(trials,'emg_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        emg_acc_plot.savefig(os.path.join(resultpath,'emgOnly_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'emgOnly_acc_compare.png'))
        emg_acc_box.savefig(os.path.join(resultpath,'emgOnly_box.png'))
        
        per_emgmodel=boxplot_param(table_readable,'emg model','emg_mean_acc',showplot=showplot_toggle)
        per_emgmodel.savefig(os.path.join(resultpath,'emgOnly_model.png'))
        
    elif architecture=='just_eeg':
        eeg_acc_plot=plot_stat_in_time(trials,'eeg_mean_acc',showplot=showplot_toggle)
        acc_compare_plot=plot_multiple_stats_with_best(trials,['eeg_mean_acc'],runbest='eeg_mean_acc',showplot=showplot_toggle)  
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['eeg_mean_acc','mean_train_acc'],runbest='eeg_mean_acc',showplot=showplot_toggle)
        eeg_acc_box=scatterbox(trials,'eeg_accs',showplot=showplot_toggle)
        
        '''saving figures of performance over time'''
        eeg_acc_plot.savefig(os.path.join(resultpath,'eegOnly_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'eegOnly_acc_compare.png'))
        eeg_acc_box.savefig(os.path.join(resultpath,'eegOnly_box.png'))
        
        per_eegmodel=boxplot_param(table_readable,'eeg model','eeg_mean_acc',showplot=showplot_toggle)
        per_eegmodel.savefig(os.path.join(resultpath,'eegOnly_model.png'))
    
    else:
    
        emg_acc_plot=plot_stat_in_time(trials, 'emg_mean_acc',showplot=showplot_toggle)
        eeg_acc_plot=plot_stat_in_time(trials, 'eeg_mean_acc',showplot=showplot_toggle)
        fus_acc_plot=plot_stat_in_time(trials,'fusion_mean_acc',showplot=showplot_toggle)
        
        acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','eeg_mean_acc','fusion_mean_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)
        # BELOW IF REPORTING TRAIN ACCURACY
        #acc_compare_plot=plot_multiple_stats_with_best(trials,['emg_mean_acc','eeg_mean_acc','fusion_mean_acc','mean_train_acc'],runbest='fusion_mean_acc',showplot=showplot_toggle)

        emg_acc_box=scatterbox(trials,'emg_accs',showplot=showplot_toggle)
        eeg_acc_box=scatterbox(trials,'eeg_accs',showplot=showplot_toggle)
        fus_acc_box=scatterbox(trials,'fusion_accs',showplot=showplot_toggle)
        
    
        '''saving figures of performance over time'''
        emg_acc_plot.savefig(os.path.join(resultpath,'emg_acc.png'))
        eeg_acc_plot.savefig(os.path.join(resultpath,'eeg_acc.png'))
        fus_acc_plot.savefig(os.path.join(resultpath,'fusion_acc.png'))
        acc_compare_plot.savefig(os.path.join(resultpath,'acc_compare.png'))
        
        emg_acc_box.savefig(os.path.join(resultpath,'emg_box.png'))
        eeg_acc_box.savefig(os.path.join(resultpath,'eeg_box.png'))
        fus_acc_box.savefig(os.path.join(resultpath,'fusion_box.png'))
        
        
        '''figures of performance per model choice'''
        per_emgmodel=boxplot_param(table_readable,'emg model','fusion_mean_acc',showplot=showplot_toggle)
        per_eegmodel=boxplot_param(table_readable,'eeg model','fusion_mean_acc',showplot=showplot_toggle)
        per_fusalg=boxplot_param(table_readable,'fusion algorithm','fusion_mean_acc',showplot=showplot_toggle)
        
        per_emgmodel.savefig(os.path.join(resultpath,'emg_model.png'))
        per_eegmodel.savefig(os.path.join(resultpath,'eeg_model.png'))
        per_fusalg.savefig(os.path.join(resultpath,'fus_alg.png'))
        
         
        
    
    raise KeyboardInterrupt('ending execution here!')
                                    
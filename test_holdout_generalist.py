# -*- coding: utf-8 -*-
"""

@author: pritcham

script to test specific modelling configurations, previously determined through CASH
optimisation, on the unseen Holdout dataset, in Generalist (subject-independent) setting

"""

import testFusion as fuse
import handleML as ml
import handleFeats as feats
import params as params
import pickle
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from distro_across_ppts import plot_ppt_distro, plot_ppt_rank, plot_ppt_minmax_normalised
import scipy.stats as stats
import pandas as pd
import numpy as np
import time
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import random

def update_chosen_params(space,arch):
    ''' This function overwrites the hyperparameter dict to fixed specified values.
    
    The {arch} argument determines which architecture's configuration should be used
    (and additionally includes the benchmark derived solely from literature inferences).
    
    The values below are from the results of CASH optimisation in our experiments.
    You will probably want to replace them with the results of yours.
    See the function {setup_search_space} to get an idea for the correct format.'''
    
    paramdict={
            'just_emg':{'fusion_alg':'just_emg',
                        'emg':{'emg_model_type':'LDA',
                           'LDA_solver':'eigen',
                           'shrinkage':0.07440592720562522,
                           },
                      },
            'just_eeg':{'fusion_alg':'just_eeg',
                        'eeg':{'eeg_model_type':'LDA',
                           'LDA_solver':'lsqr',
                           'shrinkage':0.043549089484270026,
                           },
                      },    
            'decision':{'fusion_alg':'svm',
                        'svmfuse':{'svm_C':0.05380895830748056,},
                        'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'eigen',
                               'shrinkage':0.3692791355027271,},
                        'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'lsqr',
                               'shrinkage':0.23494163949824712,},
                        'stack_distros':True,
                        },
            'hierarchical':{'fusion_alg':'hierarchical',
                       'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.09831710168084823,},
                       'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.9411890733402433,},
                           'stack_distros':True,
                           },
            'hierarchical_inv':{'fusion_alg':'hierarchical_inv',
                       'eeg':{'eeg_model_type':'LDA',
                               'LDA_solver':'svd',
                               'shrinkage':0.007751611941250992,},
                        'emg':{'emg_model_type':'LDA',
                               'LDA_solver':'lsqr',
                               'shrinkage':0.02289551604694198,},
                           'stack_distros':True,
                           },
            'feat_sep':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'eigen',
                             'shrinkage':0.023498953661387587,
                             },
                        'featfuse_sel_feats_together':False,
                        },
            'feat_join':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'lsqr',
                             'shrinkage':0.18709819935238686,
                             },
                        'featfuse_sel_feats_together':True,
                        },
            'lit_default_generalist':{'fusion_alg':'mean',
                                'eeg':{'eeg_model_type':'LDA',
                                       'LDA_solver':'svd',
                                       'shrinkage':None,
                                       },
                                'emg':{'emg_model_type':'SVM_PlattScale',
                                       'kernel':'linear',
                                       'svm_C':1.0,
                                       'gamma':None,
                                       },
                                'stack_distros':True,
                                },
        }
    space.update(paramdict[arch])
    return space


def fuse_LOO(emg_others,eeg_others,emg_ppt,eeg_ppt,args):
    start=time.time()
    
    if args['scalingtype']:
        emg_others,emgscaler=feats.scale_feats_train(emg_others,args['scalingtype'])
        eeg_others,eegscaler=feats.scale_feats_train(eeg_others,args['scalingtype'])
        emg_ppt=feats.scale_feats_test(emg_ppt,emgscaler)
        eeg_ppt=feats.scale_feats_test(eeg_ppt,eegscaler)
        
    if args['fusion_alg'] in ['svm','lda','rf']:
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_metamodel(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
   
    elif args['fusion_alg']=='hierarchical':
        
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical(emg_others, eeg_others, emg_ppt, eeg_ppt, args)

    elif args['fusion_alg']=='hierarchical_inv':          
                                              
        if not args['get_train_acc']:    
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.fusion_hierarchical_inv(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
             
    elif args['fusion_alg']=='featlevel':  
                        
        targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.feature_fusion(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
    elif args['fusion_alg']=='just_emg':
        
        if not args['get_train_acc']:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EMG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
    
    elif args['fusion_alg']=='just_eeg':
        
        if not args['get_train_acc']:    
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels=fuse.only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
        else:
            targets, predlist_emg, predlist_eeg, predlist_fusion, classlabels, traintargs, predlist_train=fuse.only_EEG(emg_others, eeg_others, emg_ppt, eeg_ppt, args)
                
    else:
                    
        emg_others=ml.drop_ID_cols(emg_others)
        eeg_others=ml.drop_ID_cols(eeg_others)
        
        pptidx=int(emg_ppt['ID_pptID'].iloc[0])-1
        sel_cols_eeg=[eeg_others.columns.get_loc(col) for col in args['eeg_feats_LOO'].iloc[pptidx].tolist()]
        sel_cols_emg=[emg_others.columns.get_loc(col) for col in args['emg_feats_LOO'].iloc[pptidx].tolist()]
            
        eeg_others=eeg_others.iloc[:,sel_cols_eeg]
        emg_others=emg_others.iloc[:,sel_cols_emg]
        
        emg_model,eeg_model=fuse.train_models_opt(emg_others,eeg_others,args)
    
        classlabels = emg_model.classes_
        
        emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        eeg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
            
        targets, predlist_emg, predlist_eeg, predlist_fusion,_,_,_= fuse.refactor_synced_predict(emg_ppt, eeg_ppt, emg_model, eeg_model, classlabels,args, sel_cols_eeg,sel_cols_emg)

    gest_truth,gest_pred_emg,gest_pred_eeg,gest_pred_fusion,gesturelabels=fuse.classes_from_preds(targets,predlist_emg,predlist_eeg,predlist_fusion,classlabels)
        
    emg_acc=(fuse.accuracy_score(gest_truth,gest_pred_emg))
    eeg_acc=(fuse.accuracy_score(gest_truth,gest_pred_eeg))
    acc=(fuse.accuracy_score(gest_truth,gest_pred_fusion))
    
    kappa=(fuse.cohen_kappa_score(gest_truth,gest_pred_fusion))
    
    if args['get_train_acc']:
        train_truth=[params.idx_to_gestures[gest] for gest in traintargs]
        train_preds=[params.idx_to_gestures[pred] for pred in predlist_train]
        train_acc=(fuse.accuracy_score(train_truth,train_preds))
    else:
        train_acc=(0)

    end=time.time()
    
    if 'getPreds' in args:
        if args['getPreds']==True:
            return {
                'loss': 1-acc,
                'kappa':kappa,
                'fusion_acc':acc,
                'emg_acc':emg_acc,
                'eeg_acc':eeg_acc,
                'train_acc':train_acc,
                'elapsed_time':end-start,
                'gest_truth':gest_truth,
                'gest_preds':gest_pred_fusion,}
    return {
        'loss': 1-acc,
        'kappa':kappa,
        'fusion_acc':acc,
        'emg_acc':emg_acc,
        'eeg_acc':eeg_acc,
        'train_acc':train_acc,
        'elapsed_time':end-start,}




if __name__ == '__main__':
        
    test_archs=False
    
    test_litDefault=False
    
    #load and balance Development Set with which to train models
    trainEEGpath=params.jeong_eeg_noholdout
    trainEMGpath=params.jeong_emg_noholdout
    
    trainEEG=pd.read_csv(trainEEGpath)
    trainEMG=pd.read_csv(trainEMGpath)
    trainEMG,trainEEG=fuse.balance_set(trainEMG,trainEEG)
    
    #define paths for Holdout subjects' datasets
    ppt1={'emg_path':params.holdout_1_emg,
          'eeg_path':params.holdout_1_eeg}
    ppt6={'emg_path':params.holdout_6_emg,
          'eeg_path':params.holdout_6_eeg}
    ppt11={'emg_path':params.holdout_11_emg,
          'eeg_path':params.holdout_11_eeg}
    ppt16={'emg_path':params.holdout_16_emg,
          'eeg_path':params.holdout_16_eeg}
    ppt21={'emg_path':params.holdout_21_emg,
          'eeg_path':params.holdout_21_eeg}
    
    holdout_ppts=[ppt1,ppt6,ppt11,ppt16,ppt21]
    
    
    
    if test_litDefault:
        
        space=fuse.setup_search_space('decision',include_svm=True)
        space=update_chosen_params(space, 'lit_default_generalist')
        
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_lit_def=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        # save results
        rootpath=params.holdout_generalist_result_dir
        with open(os.path.join(rootpath,"lit_def_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_lit_def,f)
        
        ppt_scores_lit_def.reset_index(drop=False)
        ppt_scores_lit_def.to_csv(params.holdout_generalist_litDefault)
        
            
    
    if test_archs:
        
        ''' UNIMODAL EEG '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('just_eeg',include_svm=False)
        space=update_chosen_params(space,'just_eeg')
        space.update({'trialmode':'LOO'})
        # load pre-identified arrays of selected features
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_just_eeg=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        
        
        ''' UNIMODAL EMG '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('just_emg',include_svm=False)        
        space=update_chosen_params(space,'just_emg')
        space.update({'trialmode':'LOO'})
        
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_just_emg=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
            
        
        
        
        ''' FEATURE-LEVEL FUSION, JOINT SELECTION '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('featlevel',include_svm=False)
        
        space=update_chosen_params(space,'feat_join')
        space.update({'trialmode':'LOO'})
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        emgeegcols=pd.read_csv(params.jointemgeegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,
                      'jointemgeeg_feats_LOO':emgeegcols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_feat_join=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        
        
        
        ''' FEATURE-LEVEL FUSION, SEPARATE SELECTION '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('featlevel',include_svm=False)
        space=update_chosen_params(space,'feat_sep')
        space.update({'trialmode':'LOO'})
        
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_feat_sep=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        
        
        
        ''' DECISION-LEVEL FUSION '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('decision',include_svm=False)
        space=update_chosen_params(space,'decision')
        space.update({'trialmode':'LOO'})
        
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_dec=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        
        
        
        
        ''' HIERARCHICAL FUSION '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('hierarchical',include_svm=False)        
        space=update_chosen_params(space,'hierarchical')
        space.update({'trialmode':'LOO'})
        
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_hierarch=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
        
        
        
        ''' INVERSE HIERARCHICAL FUSION '''
        # create appropriate model definition & update with specific config
        space=fuse.setup_search_space('hierarchical_inv',include_svm=False)        
        space=update_chosen_params(space,'hierarchical_inv')
        space.update({'trialmode':'LOO'})
        
        emg_cols=pd.read_csv(params.emgLOOfeatpath,delimiter=',',header=None)
        eeg_cols=pd.read_csv(params.eegLOOfeatpath,delimiter=',',header=None)
        space.update({'emg_feats_LOO':emg_cols,
                      'eeg_feats_LOO':eeg_cols,})
        
        ppt_scores=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=fuse_LOO(trainEMG,trainEEG,emg,eeg,space)
            ppt_scores.append(results)
            
        ppt_scores_inv_hierarch=pd.DataFrame(ppt_scores, index=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        
    
    
        # save results
        rootpath=params.holdout_generalist_result_dir
        with open(os.path.join(rootpath,"emg_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_just_emg,f)
        with open(os.path.join(rootpath,"eeg_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_just_eeg,f)
        with open(os.path.join(rootpath,"decision_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_dec,f)
        with open(os.path.join(rootpath,"featsep_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_feat_sep,f)
        with open(os.path.join(rootpath,"featjoint_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_feat_join,f)
        with open(os.path.join(rootpath,"hierarch_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_hierarch,f)
        with open(os.path.join(rootpath,"inv_hierarch_scores.pkl"),'wb') as f:
            pickle.dump(ppt_scores_inv_hierarch,f)

        
        ppt_scores_all=[ppt_scores_just_eeg,ppt_scores_just_emg,ppt_scores_dec,ppt_scores_feat_sep,ppt_scores_feat_join,ppt_scores_hierarch,ppt_scores_inv_hierarch]
        ppt_scores_all=pd.concat(ppt_scores_all,axis=0,keys=['just_eeg','just_emg','decision','feat_sep','feat_join','hierarch','inv_hierarch'])
        ppt_scores_all.index.names=['arch','ppt']
        
        ppt_scores_all.reset_index(drop=False)
        ppt_scores_all.to_csv(params.holdout_generalist_full_results)
    


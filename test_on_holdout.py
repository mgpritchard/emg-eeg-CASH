# -*- coding: utf-8 -*-
"""

@author: pritcham

script to test specific modelling configurations, previously determined through CASH
optimisation, on the unseen Holdout dataset, in Bespoke (subject-specific) setting

"""

import testFusion as fuse
import handleML as ml
import params as params
import pickle
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def update_chosen_params(space,arch):
    ''' This function overwrites the hyperparameter dict to fixed specified values.
    
    The {arch} argument determines which architecture's configuration should be used
    (and additionally includes the benchmark derived solely from literature inferences).
    
    The values below are from the results of CASH optimisation in our experiments.
    You will probably want to replace them with the results of yours.
    See the function {setup_search_space} to get an idea for the correct format.'''
    
    paramdict={
            'just_emg':{'fusion_alg':'just_emg',
                        'emg':{'emg_model_type':'SVM_PlattScale',
                           'kernel':'rbf',
                           'svm_C':4.172505640055673,
                           'gamma':0.012556011834910268,
                           },
                      },
            'just_eeg':{'fusion_alg':'just_eeg',
                        'eeg':{'eeg_model_type':'LDA',
                           'LDA_solver':'lsqr',
                           'shrinkage':0.037969462143491395,
                           },
                      },
            'decision':{'fusion_alg':'highest_conf',
                        'eeg':{'eeg_model_type':'RF',
                               'n_trees':85,
                               'max_depth':5,},
                        'emg':{'emg_model_type':'SVM_PlattScale',
                               'kernel':'rbf',
                               'svm_C':98.91885185586297,
                               'gamma':0.013119782396855456,},
                        'stack_distros':True,
                        },
            'feat_sep':{'fusion_alg':'featlevel',
                        'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.6653418849680925,
                             #shrinkage not actually relevant to SVD
                             #this is just to keep the code happy
                             },
                        'featfuse_sel_feats_together':False,
                        },
            'feat_join':{'fusion_alg':'featlevel',
                         'featfuse':
                            {'featfuse_model_type':'LDA',
                             'LDA_solver':'svd',
                             'shrinkage':0.5048542532123359,
                             #shrinkage not actually relevant to SVD
                             #this is just to keep the code happy
                             },
                        'featfuse_sel_feats_together':True,
                        },
            'hierarchical':{'fusion_alg':'hierarchical',
                           'eeg':{'eeg_model_type':'QDA',
                                  'regularisation':0.4558963480892469,},
                           'emg':{'emg_model_type':'SVM_PlattScale',
                                  'kernel':'rbf',
                                  'svm_C':19.403739187394663,
                                  'gamma':0.013797650887036847,},
                           'stack_distros':True,
                           },
            'hierarchical_inv':{'fusion_alg':'hierarchical_inv',
                           'eeg':{'eeg_model_type':'RF',
                                  'n_trees':75,
                                  'max_depth':5,},
                           'emg':{'emg_model_type':'QDA',
                                  'regularisation':0.3324563281128364,},
                           'stack_distros':True,
                           },
            'lit_default_bespoke':{'fusion_alg':'svm',
                                'svmfuse':{'fusesvmPlatt':True,
                                    'svm_C':1.0,
                                    'kernel':'rbf',
                                    'gamma':'scale',},
                                'eeg':{'eeg_model_type':'LDA',
                                       'LDA_solver':'svd',
                                       'shrinkage':None,
                                       },
                                'emg':{'emg_model_type':'SVM_PlattScale',
                                       'kernel':'rbf',
                                       'svm_C':1.0,
                                       'gamma':'scale',
                                       },
                                'stack_distros':True,
                                },
        }
    space.update(paramdict[arch])
    return space


def test_system(arch,emg,eeg):
    
    #create a model definition space appropriate to the architecture
    if arch in ['just_emg','just_eeg','decision','hierarchical','hierarchical_inv']:
        space=fuse.setup_search_space(arch,include_svm=True)
    elif arch in ['feat_sep','feat_join']:
        space=fuse.setup_search_space('featlevel',include_svm=True)
        if arch=='feat_sep':
            space.update({'featfuse_sel_feats_together':False})
        elif arch=='feat_join':
            space.update({'featfuse_sel_feats_together':True})
    elif arch== 'lit_default_bespoke':
        space=fuse.setup_search_space('decision',include_svm=True)
    else: raise(ValueError(('Unknown architecture: '+arch)))
    
    #update model definition space with relevant variables
    space.update({'emg_set':emg,'eeg_set':eeg,'data_in_memory':True,'prebalanced':True,'trialmode':'WithinPpt','l1_sparsity':0.005,'l1_maxfeats':40})
    
    #update the model definition space with specific CASH-identified configuration
    space = update_chosen_params(space,arch)
    
    #perform classification and return results
    result=fuse.function_fuse_withinppt(space)
    result.update({'arch':arch})
    return result
    

def test_chosen_bespokes(emg,eeg):
    #testing all possible unimodal and multimodal architectures
    archs=['just_emg','just_eeg','decision','feat_sep','feat_join','hierarchical','hierarchical_inv']
    results=[]
    for arch in archs:
        for n in range(100):
            results.append(pd.DataFrame(test_system(arch,emg,eeg)))
    results=pd.concat(results).set_index('arch')
    return results


def test_lit_def_bespoke(emg,eeg):
    #separate function to test the literature-derived classical ML benchmark
    archs=['lit_default_bespoke']
    results=[]
    for arch in archs:
        for n in range(100):
            results.append(pd.DataFrame(test_system(arch,emg,eeg)))
    results=pd.concat(results).set_index('arch')
    return results



if __name__ == '__main__':
    test_lit_defaults=False
    test_archs=False
    
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
    
    
    if test_lit_defaults:
        
        ppt_scores_list=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=test_lit_def_bespoke(emg,eeg)
            ppt_scores_list.append(results)
        ppt_scores=pd.concat(ppt_scores_list, axis=0, keys=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        ppt_scores=ppt_scores.swaplevel(-2,-1)
        ppt_scores.index.names=['arch','ppt']
        
        # save results
        ppt_scores.reset_index(drop=False)
        ppt_scores.to_csv(params.holdout_bespoke_100repeats_litDefault)
    
    
    
    if test_archs:
        
        ppt_scores_list=[]
        for ppt in holdout_ppts:
            emg=pd.read_csv(ppt['emg_path'],delimiter=',')
            eeg=pd.read_csv(ppt['eeg_path'],delimiter=',')
            emg,eeg=fuse.balance_set(emg,eeg)
            results=test_chosen_bespokes(emg,eeg)
            ppt_scores_list.append(results)
        ppt_scores=pd.concat(ppt_scores_list, axis=0, keys=['ppt1','ppt6','ppt11','ppt16','ppt21'])
        ppt_scores=ppt_scores.swaplevel(-2,-1)
        ppt_scores.index.names=['arch','ppt']
        
        # save results
        ppt_scores.reset_index(drop=False)
        ppt_scores.to_csv(params.holdout_bespoke_100repeats_results)
    
    
    
    
# -*- coding: utf-8 -*-
"""

@author: pritcham

module for OS-dependent parameters

"""

from sys import platform
import os

gestures_to_idx = {'close':1.,'open':2.,'grasp':3.,'lateral':4.,'tripod':5.,'neutral':0.,'cylindrical':6.,'spherical':7.,'lumbrical':8.,'rest':9.}
idx_to_gestures = {1.:'close',2.:'open',3.:'grasp',4.:'lateral',5.:'tripod',0.:'neutral',6.:'cylindrical',7.:'spherical',8.:'lumbrical',9.:'rest'}

gestures_to_idx_binary = {'neutral':0.,'cylindrical':1.,'spherical':1.,'lumbrical':1.,'rest':0.}
idx_to_gestures_binary = {1.:'grasp',0.:'rest'}

gestures_to_idx_deeplearn = {'cylindrical':0.,'lumbrical':1.,'rest':2.,'spherical':3.}
idx_to_gestures_deeplearn = {0.:'cylindrical',1.:'lumbrical',2.:'rest',3.:'spherical'}

currentpath=os.path.dirname(__file__)

if platform == 'darwin':
    raise ValueError('On your own, sorry')
    
    
# data files after MATLAB pre-processing
    
jeong_EMGdir='H:/Jeong11tasks_data/EMG/CSVs/' 
jeong_EEGdir='H:/Jeong11tasks_data/EEG/CSVs/'  

# processed featuresets  
    
jeong_EMGfeats='H:/Jeong11tasks_data/jeong_EMGfeats.csv'   
jeong_EEGfeats='H:/Jeong11tasks_data/EEGnoCSP_WidebandFeats.csv' 

# featuresets divided into Holdout and Development

holdout_1_emg=jeong_EMGfeats[-4]+'_ppt1.csv'
holdout_1_eeg=jeong_EEGfeats[-4]+'_ppt1.csv'
holdout_6_emg=jeong_EMGfeats[-4]+'_ppt6.csv'
holdout_6_eeg=jeong_EEGfeats[-4]+'_ppt6.csv'
holdout_11_emg=jeong_EMGfeats[-4]+'_ppt11.csv'
holdout_11_eeg=jeong_EEGfeats[-4]+'_ppt11.csv'
holdout_16_emg=jeong_EMGfeats[-4]+'_ppt16.csv'
holdout_16_eeg=jeong_EEGfeats[-4]+'_ppt16.csv'
holdout_21_emg=jeong_EMGfeats[-4]+'_ppt21.csv'
holdout_21_eeg=jeong_EEGfeats[-4]+'_ppt21.csv'

emg_no_holdout=jeong_EMGfeats[-4]+'_noHO.csv'
eeg_no_holdout=jeong_EEGfeats[-4]+'_noHO.csv'

# option to stick the very large featuresets in a different location if deploying remotely

jeong_EMGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEMG_noholdout.csv')
jeong_EEGnoHO_server=os.path.join(currentpath,'lit_data_expts/jeong/datasets/jeongEEG_noholdout.csv')



# assorted files capturing reduced feature sets for leave-one-subject-out Generalist classification
# these take a long time to generate, so we do it once across all experiments rather than repeating in each CASH iteration

joint_LOO_feat_idx_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/joint_DevLOO_featIDXs.csv')
joint_LOO_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/joint_DevLOO_featsels.csv')

eeg_LOO_feat_idx_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/eeg_DevLOO_featIDXs.csv')
eeg_LOO_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/eeg_DevLOO_featsels.csv')

emg_LOO_feat_idx_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/emg_DevLOO_featIDXs.csv')
emg_LOO_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/emg_DevLOO_featsels.csv')

joint_LOO_HOtest_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/joint_FullDevforHO_featsels.csv')
eeg_LOO_HOtest_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/eeg_FullDevforHO_featsels.csv')
emg_LOO_HOtest_feats_csv=os.path.join(currentpath,'lit_data_expts/jeong/feat_selections/emg_FullDevforHO_featsels.csv')

emgLOOfeatsBlankRows=emg_LOO_feats_csv[-4]+'_blankRows.csv'
eegLOOfeatsBlankRows=eeg_LOO_feats_csv[-4]+'_blankRows.csv'
jointLOOfeatsBlankRows=joint_LOO_feats_csv[-4]+'_blankRows.csv'

emgLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/feat_selections/emg_LOO_feats_15pct.csv')
eegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/feat_selections/eeg_LOO_feats_L1_88.csv')
jointemgeegLOOfeatpath=os.path.join(currentpath, 'lit_data_expts/jeong/feat_selections/joint_LOO_feats_L1_176.csv')



# directory for results and specific paths for results of Holdout testing

jeong_results_dir='lit_data_expts/jeong/results/'

holdout_bespoke_100repeats_litDefault=os.path.join(currentpath,'lit_data_expts/jeong/results/bespoke_HO_litDefault_100reps.csv')
holdout_bespoke_100repeats_results=os.path.join(currentpath,'lit_data_expts/jeong/results/bespoke_HO_results_100reps.csv')

holdout_generalist_result_dir=os.path.join(currentpath,'lit_data_expts/jeong/results/generalist/HO/')
holdout_generalist_full_results=os.path.join(currentpath,'lit_data_expts/jeong/results/generalist_HO_results.csv')
holdout_generalist_litDefault=os.path.join(currentpath,'lit_data_expts/jeong/results/generalist_HO_litDefault.csv')
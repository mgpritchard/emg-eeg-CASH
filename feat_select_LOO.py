#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pritcham

"""
import numpy as np
import pandas as pd
import params
import testFusion as fuse

def inject_blank_rows(unadjusted_path, adjusted_path, nfeats=88):
    feats=pd.read_csv(unadjusted_path,header=None)
    nulls=['N/A']*(nfeats+1)
    nullrow=pd.DataFrame(np.array(nulls).reshape(-1,len(nulls)),columns=feats.columns)
    feats2=pd.concat([feats[:0],nullrow,feats[0:]]).reset_index(drop=True)
    feats2=pd.concat([feats2[:5],nullrow,feats2[5:]]).reset_index(drop=True)
    feats2=pd.concat([feats2[:10],nullrow,feats2[10:]]).reset_index(drop=True)
    feats2=pd.concat([feats2[:15],nullrow,feats2[15:]]).reset_index(drop=True)
    feats2=pd.concat([feats2[:20],nullrow,feats2[20:]]).reset_index(drop=True)
    feats2.to_csv(adjusted_path,header=False,index=False)
    
def insert_holdout_featrows(blankrows_path,holdout_path,final_path):
    feats=pd.read_csv(blankrows_path,header=None)
    feats_20=pd.read_csv(holdout_path,header=None).T
    feats.loc[0]=feats_20.loc[0]
    feats.loc[5]=feats_20.loc[0]
    feats.loc[10]=feats_20.loc[0]
    feats.loc[15]=feats_20.loc[0]
    feats.loc[20]=feats_20.loc[0]
    feats.to_csv(final_path,header=False,index=False)


if __name__ == '__main__':
    args=fuse.setup_search_space(architecture='decision',incl_svm=False)
    fuse.get_LOO_feats(args,jointly=False)
    
    fuse.make_featsel_all20(joint=False)
    
    
    args=fuse.setup_search_space(architecture='featlevel',incl_svm=False)
    fuse.get_LOO_feats(args,jointly=True)
    
    fuse.make_featsel_all20(joint=True)
    
    
    unadjusted_path=params.emg_LOO_feats_csv
    blankrows_path=params.emgLOOfeatsBlankRows
    inject_blank_rows(unadjusted_path, blankrows_path, nfeats=88)
    
    holdout_path=params.emg_LOO_HOtest_feats_csv
    final_path=params.emgLOOfeatpath
    insert_holdout_featrows(blankrows_path,holdout_path,final_path)
    
    
    unadjusted_path=params.eeg_LOO_feats_csv
    blankrows_path=params.eegLOOfeatsBlankRows
    inject_blank_rows(unadjusted_path, blankrows_path, nfeats=88)
    
    holdout_path=params.eeg_LOO_HOtest_feats_csv
    final_path=params.eegLOOfeatpath
    insert_holdout_featrows(blankrows_path,holdout_path,final_path)
    
    
    unadjusted_path=params.joint_LOO_feats_csv
    blankrows_path=params.jointLOOfeatsBlankRows
    inject_blank_rows(unadjusted_path, blankrows_path, nfeats=88*2)
    
    holdout_path=params.joint_LOO_HOtest_feats_csv
    final_path=params.jointemgeegLOOfeatpath
    insert_holdout_featrows(blankrows_path,holdout_path,final_path)
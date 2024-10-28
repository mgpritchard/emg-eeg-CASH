# emg-eeg-CASH
Code associated with the in-production paper "Multimodal EMG-EEG Fusion Strategies for Upper-Limb Gesture classification" [Pritchard et al.].

1) use MATLAB scripts to pre-process EMG & EEG data.
2) use feat_extract_and_holdout_split.py to generate featuresets and reserve Holdout dataset.
3) use feat_select_LOO.py to perform leave-one-subject-out feature selection in advance.
4) use testFusion.py for each fusion or unimodal architecture in each of bespoke (WithinPpt) and generalist (LOO) settings.
5) update test_on_holdout.py and test_holdout_generalist.py with CASH-identified configurations
6) use test_on_holdout.py and test_holdout_generalist.py to verify CASH-identified configurations and literature-inferred configuration on unseen Holdout subjects

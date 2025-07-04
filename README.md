# emg-eeg-CASH
Code associated with the 2025 paper "An Investigation of Multimodal EMG-EEG Fusion Strategies for Upper-Limb Gesture classification" [Pritchard et al. 2025] https://doi.org/10.1088/1741-2552/ade1f9.

1) use MATLAB scripts to pre-process EMG & EEG data. (Relies upon bbci toolbox & associated scripts from: Jeong J; Cho J; Shim K; Kwon B; Lee B; Lee D; Lee D; Lee S (2020): Supporting data for "Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions" GigaScience Database as found at https://doi.org/10.5524/100788 within Scripts.tar.gz).
3) use `feat_extract_and_holdout_split.py` to generate featuresets and reserve Holdout dataset.
4) use `feat_select_LOO.py` to perform leave-one-subject-out feature selection in advance.
5) use `testFusion.py` for each fusion or unimodal architecture in each of bespoke (`WithinPpt`) and generalist (`LOO`) settings.
6) update `test_on_holdout.py` and `test_holdout_generalist.py` with CASH-identified configurations
7) use `test_on_holdout.py` and `test_holdout_generalist.py` to verify CASH-identified configurations and literature-inferred configuration on unseen Holdout subjects

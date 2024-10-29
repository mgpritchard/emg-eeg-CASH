% -------------------------------------------------------------------------
% Adapted from ACC_HandGrasping.m of "Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions" [Jeong et al, 2021] found within Scripts.tar.gz at https://doi.org/10.5524/100788 

% Please add the "bbci toolbox from" the aforementioned GigaDB repository in a subfolder as {HERE}\Reference_toolbox\bbci_toolbox\
%--------------------------------------------------------------------------

%% Initalization
clc; close all; clear all;

% Identify (and move to) directory where converted data file downloaded
dd='{PATH_TO_DATASET}\EEG\m_files\';
cd '{PATH_TO_DATASET}\EEG\m_files\';

% Identify directory to place processed data files.
% NB: THIS SHOULD MATCH jeong_EEGdir IN params.py
csv_dir='{PATH_TO_DATASET}\EEG\CSVs\';

% Ensure bbci toolbox is added to path
addpath(genpath('{PATH_TO_HERE}\Reference_toolbox\bbci_toolbox\'))


datadir = dir('*.mat');
filelist = {datadir.name};


% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Processing
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 4, range of [2-30] Hz
    filterBank = {[2 30]};
    for filt = 1:length(filterBank)
        clear epo_check epo epoRest
        filelist{i}
        filterBank{filt}
        
        cnt = proc_filtButter(cnt, 4 ,filterBank{filt});
        epo=cntToEpo(cnt,mrk,ival);
        
        % Select channels
        epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
            'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        
        classes=size(epo.className,2);
        
        trial=50;
        
        % Set the number of rest trials to the same as the number of other classes.
        
        % THIS METHOD SETS RANDOM SAMPLE DETERMINISTICALLY BASED ON COMBO
        % OF SUBJECT AND SESSION; HENCE WILL BE THE SAME SAMPLE FOR BOTH
        % EMG AND EEG
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                %epoRest=proc_selectClasses(epo,{epo.className{ii}});
                epoRest=proc_selectClasses(epo,epo.className(ii)); %supposedly faster

                if ~(size(epoRest.x,3)==trial)  %randomness dependent on subject/session for sync purposes
                    disp('Need to downsample')

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);


                    epoRest.x=datasample(stream,epoRest.x,trial,3,'Replace',false);
                    epoRest.y=datasample(stream,epoRest.y,trial,2,'Replace',false);
                end
            else
                %epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                epo_check(ii)=proc_selectClasses(epo,epo.className(ii));

                if ~(size(epo_check(ii).x,3)==trial)
                    disp('Need to downsample')

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);
                
                     % Randomization
                    epo_check(ii).x=datasample(stream,epo_check(ii).x,trial,3,'Replace',false); %selects 50 of each class
                    epo_check(ii).y=datasample(stream,epo_check(ii).y,trial,2,'Replace',false);
                end
            end
        end

        if ~any(strcmp([epo_check(:).className],'Rest'))
            epo_check(size(epo_check,2)+1)=epoRest;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

        done_slicing=false;
        if done_slicing==false
            for j=1:length(epo_check)
                epoclass=epo_check(j);
                gesture=epoclass.className;
                % 50 trials per 3 grasps + rest per 3 sessions = 600 total
                filename=split(epoclass.file,'\');
                filename=filename{end};
                fnameparts=split(filename,'_');
                subject=strcat('00',erase(fnameparts{3},'sub'));
                session=erase(fnameparts{2},'session');
                for jj = 1:trial
                    classTable=array2table([epoclass.t',epoclass.x(:,:,jj)],'VariableNames',[{'Timestamp'},epoclass.clab]);
                    csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(jj),'.csv'));
                    csvpath=strcat(csv_dir,csvname);
                    writetable(classTable,csvpath);
                end
            end
        end
   
    end   
    
end

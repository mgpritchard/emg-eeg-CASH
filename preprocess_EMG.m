% -------------------------------------------------------------------------
% Adapted from ACC_HandGrasping.m of "Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions" [Jeong et al, 2021] found within Scripts.tar.gz at https://doi.org/10.5524/100788 

% Please add the "bbci toolbox from" the aforementioned GigaDB repository in a subfolder as {HERE}\Reference_toolbox\bbci_toolbox\
%--------------------------------------------------------------------------

%% Initalization
clc; close all; clear all;

% Identify (and move to) directory where converted data file downloaded
dd='{PATH_TO_DATASET}\EMG\EMG_ConvertedData\'; 
cd '{PATH_TO_DATASET}\EMG\EMG_ConvertedData\';

% Identify directory to place processed data files.
% NB: THIS SHOULD MATCH jeong_EMGdir IN params.py
csv_dir='{PATH_TO_DATASET}\EMG\CSVs\';

% Ensure bbci toolbox is added to path
addpath(genpath('{PATH_TO_HERE}\Reference_toolbox\bbci_toolbox\'))


datadir = dir('*.mat');
filelist = {datadir.name};
idx=contains(filelist,'grasp')&contains(filelist,'real');
filelist=filelist(idx);

% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Processing
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 5, range of [10-500] Hz
    filterBank = {[10 500]};
    for filt = 1:length(filterBank)
        clear epo_check epo epoRest
        filelist{i}
        filterBank{filt}
        
        cnt = proc_filtButter(cnt, 5 ,filterBank{filt});
        epo=cntToEpo(cnt,mrk,ival);
        epo.x=abs(epo.x); %rectifying EMG
        % Select channels 
        
        epo = proc_selectChannels(epo, {'EMG_1','EMG_2','EMG_3','EMG_4',...
            'EMG_5','EMG_6'}); %all except EMG_ref which is elbow
       

        classes=size(epo.className,2);
        
        trial=50;
        
        % Set the number of rest trial to the same as the number of other classes.
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                epoRest=proc_selectClasses(epo,{epo.className{ii}});


                if ~(size(epoRest.x,3)==trial)
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
                epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                
                if ~(size(epo_check(ii).x,3)==trial)
                    disp('Need to downsample')

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);

                    epo_check(ii).x=datasample(stream,epo_check(ii).x,trial,3,'Replace',false);
                    epo_check(ii).y=datasample(stream,epo_check(ii).y,trial,2,'Replace',false);
                end
            end
        end

        if ~any(strcmp([epo_check(:).className],'Rest'))
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        

        done_slicing=false;
        if done_slicing==false
            for j=1:length(epo_check)
                epoclass=epo_check(j);
                gesture=epoclass.className;
                % 50 trials per 3 grasps per 3 sessions = 450 total
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

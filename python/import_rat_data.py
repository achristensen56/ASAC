import numpy as np
import scipy.io as sio
import glob
from sklearn import linear_model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from itertools import cycle
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from scipy.stats import sem
from scipy import stats
from scipy import ndimage
from matplotlib import cm
import pickle


color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])

import warnings
warnings.simplefilter("ignore")


import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'



color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])



def get_frame_data(input_file):

    frame_data = pd.read_csv(input_file)
    ms_sig = frame_data[' miniscope_sync'].values
    frame_data_diff = frame_data.diff()
    time_diff = frame_data_diff['Time[s]'].values
    for ind in range(len(time_diff)):
        if frame_data_diff[' miniscope_sync'][ind] == 1:
            if time_diff[ind] < 0.0001:
                ms_sig[ind]=0

    frame_data_diff = frame_data.diff()
    time = frame_data['Time[s]'].values
    ms_sig = frame_data[' miniscope_sync'].values
    st_sig = frame_data[' new_trial'].values
    cnp_sig = frame_data[' center_poke_start'].values
    stim_sig = frame_data[' stimulus'].values
    resp_sig = frame_data[' response'].values

    try:
        behav_sig = frame_data[' behavior camera'].values
        behav_sig = 1- behav_sig # flip 0s and 1s
        print('behavior camera sync signal found and loaded')
        behave_camera_exist = 1
        plt.plot(time, 2*ms_sig,'x-')
        plt.plot(time,behav_sig,'.-')
        plt.xlim([3000,3001])


        plt.xlim([3000,3001])
    except:
        print('no behavior camera sync data found')
        behave_camera_exist = 0


    frame_num = frame_data_diff.replace({-1.0:0}).cumsum()[' miniscope_sync'].values
    print('the number of frames according to the sync signal is:', frame_num[-1])

    def find_pulses(time,array,frame_num,min_pulse_dur=1):

        rise_inds = np.where(np.diff(array)==1)[0]+1
        fall_inds = np.where(np.diff(array)==-1)[0]+1

        rise_times = time[rise_inds]
        fall_times = time[fall_inds]

        pulse_durs = (fall_times-rise_times)*1000

        true_pulse_inds = np.where(pulse_durs>min_pulse_dur)

        rise_inds = rise_inds[true_pulse_inds]
        fall_inds = fall_inds[true_pulse_inds]
        rise_times = rise_times[true_pulse_inds]
        fall_times = fall_times[true_pulse_inds]
        pulse_durs = pulse_durs[true_pulse_inds]

        start_pulse_frame = frame_num[rise_inds]
        end_pulse_frame = frame_num[fall_inds]


        return {'rise_inds':rise_inds, 'fall_inds':fall_inds, 'rise_times':rise_times,
                'fall_times':fall_times, 'pulse_durs':pulse_durs,
                'start_pulse_frame':start_pulse_frame, 'end_pulse_frame':end_pulse_frame}



    # FIND INITIAL CENTER NOSEPOKE FRAMES
    start_poke_frames = find_pulses(time,cnp_sig,frame_num)['start_pulse_frame']
    start_poke_inds = find_pulses(time,cnp_sig,frame_num)['rise_inds']


    # FIND POTENTIAL STIM STARTS
    stim_start_frames = find_pulses(time,stim_sig,frame_num)['start_pulse_frame']
    stim_start_inds = find_pulses(time,stim_sig,frame_num)['rise_inds']

    # FIND POTENTIAL STIM ENDS
    stim_end_frames = find_pulses(time,stim_sig,frame_num)['end_pulse_frame']
    stim_end_inds = find_pulses(time,stim_sig,frame_num)['fall_inds']

    # FIND POTENTIAL RESPONSE PHASE TIMES
    response_frames = find_pulses(time,resp_sig,frame_num)['start_pulse_frame']
    response_inds = find_pulses(time,resp_sig,frame_num)['rise_inds']
    print(len(response_inds))

    # IF BEHAVIOR CAMERA SIGNAL EXISTS FIND THE FRAMES
    if behave_camera_exist:
        behave_frames = find_pulses(time,behav_sig,frame_num,min_pulse_dur=1)['start_pulse_frame']
        behave_inds = find_pulses(time,behav_sig,frame_num,min_pulse_dur=1)['rise_inds']
        print(find_pulses(time,behav_sig,frame_num)['rise_inds']-find_pulses(time,behav_sig,frame_num,min_pulse_dur=1)['rise_times'])
        print(len(behave_inds))

    # GET FRAMES FOR EVERY TRIAL
    ssf = []
    sef = []
    rf = []
    for sp in start_poke_frames:
        try:
            ssf.append(stim_start_frames[np.where(stim_start_frames>sp)][0])
        except:
            ssf.append(np.nan)
        try:
            sef.append(stim_end_frames[np.where(stim_end_frames>sp)][0])
        except:
            sef.append(np.nan)
        try:
            rf.append(response_frames[np.where(response_frames>sp)][0])
        except:
            rf.append(np.nan)


    ssf = np.array(ssf)
    sef = np.array(sef)
    rf = np.array(rf)

    for ind in range(len(ssf)-1):
        if ssf[ind]>start_poke_frames[ind+1]:
            ssf[ind]=np.nan
        if sef[ind]>start_poke_frames[ind+1]:
            sef[ind]=np.nan
        if rf[ind]>start_poke_frames[ind+1]:
            rf[ind]=np.nan

    if behave_camera_exist:
        return {'start_poke_frames': start_poke_frames,'stim_start_frames':ssf,'stim_end_frames':sef,'response_frames':rf,'behavior_camera_frames':behave_frames}
    else:
        return {'start_poke_frames': start_poke_frames,'stim_start_frames':ssf,'stim_end_frames':sef,'response_frames':rf}


def get_behavior_data(input_file):
    file = open(input_file,'rb')
    # dump information to that file
    session_data = pickle.load(file)

    # close the file
    file.close()

    final_ind = np.argmax(session_data['coherence']==0)
    nt=session_data['info']['num_trials']

    for key,value in session_data.items():
        if len(session_data[key])==nt:
            session_data[key]=value[:final_ind]


    temp_dir = np.array([x =='right' for x in session_data['stim_dir']])*1
    temp_dir[temp_dir==0]=-1

    temp_dict = {'coherence':session_data['coherence']*temp_dir,
    'prior_right':session_data['rightward_prior'],
    'response_right':np.array([x =='right' for x in session_data['response_side']])*1,
    'stim_right':np.array([x =='right' for x in session_data['stim_dir']])*1,
    'trial_completed':session_data['was_completed'],
    'was_correct':session_data['was_correct']}


    return pd.DataFrame.from_dict(temp_dict)


def get_interp_traces(trace_data,frame_data):

    data = pd.DataFrame.from_dict({'start_poke_frame':frame_data['start_poke_frames'], 'stim_start_frame':frame_data['stim_start_frames'],
                                    'stim_end_frame':frame_data['stim_end_frames'], 'reward_frame':frame_data['response_frames']})

    # DO INTERPOLATION
    trace_data = stats.mstats.zscore(trace_data,axis=1)

    # pre-nosepoke to end of nosepoke (500 ms before nosepoke to 500 ms after nosepoke; 10 frames total)
    pt_start_inds = np.array(data.start_poke_frame.tolist())-5 # 500 ms before nosepoke
    pt_frames = np.tile(np.array(pt_start_inds),(10,1)).T+np.tile(np.arange(10),(len(pt_start_inds),1)) # the 500 ms before the nosepoke
    pt_traces = trace_data[:,pt_frames.astype(int)]

    # stimulus (10 frames total)
    stim_start_inds = np.array(data.stim_start_frame.tolist())
    stim_end_inds = np.array(data.stim_end_frame.tolist())
    stim_inds = [np.arange(stim_start_inds[x],stim_end_inds[x]) for x in range(len(stim_start_inds))]

    nnn=[]
    for n in range(trace_data.shape[0]):
        ttt =[]
        for t in range(len(stim_inds)):
            td = trace_data[n,stim_inds[t].astype(int)]
            ttt.append(np.interp(np.arange(0,10),np.linspace(0,10,len(td)),td)) #make the stimulus 1 second
        nnn.append(ttt)
    stim_traces = np.squeeze(np.array(nnn))


    # between stim and response (500 ms, 5 frames total)
    resp_inds = np.array(data.reward_frame.tolist())
    btw_inds = [np.arange(stim_end_inds[x],resp_inds[x]) for x in range(len(stim_end_inds))]

    nnn=[]
    for n in range(trace_data.shape[0]):
        ttt =[]
        for t in range(len(btw_inds)):
            td = trace_data[n,btw_inds[t].astype(int)]
            ttt.append(np.interp(np.arange(0,5),np.linspace(0,5,len(td)),td))
        nnn.append(ttt)
    btw_traces = np.squeeze(np.array(nnn))


    # response (10 frames total)
    resp_frames = np.tile(np.array(resp_inds),(10,1)).T+np.tile(np.arange(10),(len(resp_inds),1))
    resp_traces = trace_data[:,resp_frames.astype(int)]

    interp_traces = np.concatenate((pt_traces,stim_traces,btw_traces,resp_traces),axis=2)

    for i in range(interp_traces.shape[0]):
        for j in range(interp_traces.shape[1]):

            interp_traces[i,j,:]=ndimage.filters.gaussian_filter(interp_traces[i,j,:],1)

    return interp_traces



import scipy
import numpy as np
import scipy.io as spio


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

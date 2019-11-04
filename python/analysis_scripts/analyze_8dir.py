import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
import warnings
warnings.filterwarnings('ignore')

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

def analyze_8dir_data(filename, spikes = True, stim_start = 45, stim_end = 55, pre_start = 30, pre_end = 40, responsive_p = 0.01, rel_crit = .5):
	'''
	input file should be an 8 directions day, needs to have completed trials data, with a traces_stim_aligned field, and a stim_dir field
	which corresponds to the 8 orientation directions. 

	outputs a list for each neuron in the traces_stim_aligned file of 1. whether the neuron is reliable, 2. whether the neuron is responsive, and 
	3 for those neurons which meet the criteria specified in the function call, outputs a matrix [n_tuned_neurons 4], where 0'th column is the index 
	of the neuron, 1st column is the orientation of the max response, then the orientation selectivity index, then the direction selectivity index
	'''

	with open(filename, 'rb') as f:

		raw_data = pickle.load(f)


	completed_data = raw_data['completed_trials_data']
	data = completed_data['traces_stim_aligned'].squeeze()
	stim = completed_data['stim_dir'].squeeze()


	n_trials, n_neurons, trial_length = data.shape

	if spikes:
		sp_data = []
		for i in range(n_neurons):
			c, b1, c1, g, sn, sp, lam= constrained_foopsi(data[:, i, :].reshape(-1), p = 1)
			sp_data.append(sp > 0.0025)

		sp_data = np.array(sp_data)
		sp_data = sp_data.reshape([n_neurons, n_trials, trial_length])

		data = sp_data
	else:
		data = data.transpose([1, 0, 2]) + 1e-6




	reliable_ = []
	responsive_ = []

	#trial_vector = np.array([0]*7 + [1]*10)# + [0]*(120-50))


	resp_mat = np.empty([n_neurons, len(sorted(np.unique(stim))), 2])

	blanks_mat = np.empty([n_neurons, 2])

	for i in range(n_neurons):
		
		dff = np.mean(data[i, :, stim_start:stim_end], axis = 1)
		
		ss = np.random.choice(range(n_trials), replace = False, size = 30)
		blank_m = data[i, ss, pre_start:pre_end].mean(axis = 1)
		blank_std = np.std(data[i, ss, pre_start:pre_end])
		

		
		blanks_mat[i, 0] = np.mean(blank_m)
		blanks_mat[i, 1] = blank_std 
		
		resp_p = []
		for j, ori in enumerate(sorted(np.unique(stim))): 

			trials = np.where(stim == ori)[0]
			resp_mat[i, j, 0] = np.mean(dff[trials]) 
			resp_mat[i, j, 1] = np.std(dff[trials])
			
			f, p = ranksums(dff[trials], blank_m)
			resp_p.append(p)

		responsive_.append(np.min(resp_p))

	for i in range(n_neurons):
		umax = np.argmax(resp_mat[i, :, 0], axis = 0)
		
		um, us = resp_mat[i, umax]
		
		reliable = (um - blanks_mat[i, 0]) / (us + blanks_mat[i, 1])
			 

		reliable_.append(reliable)
		


	reliable_ = np.array(reliable_)
	responsive_ = np.array(responsive_)

	inds = np.where(np.logical_and(reliable_ > rel_crit, responsive_ < responsive_p))[0]

	#neuron, max_ori, OSI, DSI
	selectivity_mat = np.empty([len(inds), 4])

	for j, i in enumerate(inds):
		resps = resp_mat[i, :, 0]
		max_ori_ind = np.argmax(resps)

		max_ori = sorted(np.unique(stim))[max_ori_ind]

		#OSI
		ori2 = (max_ori + 180) % 360
		ori2_ind = np.where(sorted(np.unique(stim)) == ori2)

		max_resp = resps[max_ori_ind] + resps[ori2_ind]

		min_ori1 = (max_ori + 90) % 360
		min_ori2 = (max_ori - 90) % 360

		min_ind1 = np.where(sorted(np.unique(stim)) == min_ori1)
		min_ind2 = np.where(sorted(np.unique(stim)) == min_ori2)
		
		min_resp = resps[min_ind1] + resps[min_ind2]

		OSI = (max_resp - min_resp) / (max_resp + min_resp)

		max_resp = resps[max_ori_ind]
		min_resp = resps[ori2_ind]

		DSI = (max_resp - min_resp) / (max_resp + min_resp)

		selectivity_mat[j] = [i, max_ori, OSI, DSI]


	return reliable_, responsive_, selectivity_mat

def plot_rasters(n, data_to_avg, independent_variable, stim):
	fig = plt.figure(figsize=(20,20))
	ax = plt.subplot(2,2,1)
	plt.imshow(data_to_avg[np.argsort(independent_variable),n,:],aspect='auto', cmap = 'gray')
	deltas = np.where(np.diff(independent_variable[np.argsort(independent_variable)]))[0]
	for i in range(len(deltas)):
		plt.plot([20,80],[deltas[i],deltas[i]],'w--',alpha=0.4)
		#plt.arrow(30,deltas[i]-25,5*np.cos(i*np.pi/4),-5*np.sin(i*np.pi/4),color='r')
	plt.xlim([20,80])
	plt.ylim([0,len(independent_variable)])
	plt.plot([40,40],[0,len(independent_variable)],'w--',alpha=0.4)
	plt.plot([47,47],[0,len(independent_variable)],'w--',alpha=0.4)
	#ax.set_yticklabels(sorted(np.unique(stim)));
	plt.yticks(np.arange(0, len(data_to_avg), step = len(data_to_avg) / 8), sorted(np.unique(stim)))

def load_data_for_plotting(filename):
	with open(filename, 'rb') as f:
		raw_data = pickle.load(f)

	completed_data = raw_data['completed_trials_data']
	data = completed_data['traces_stim_aligned'].squeeze()
	stim = completed_data['stim_dir'].squeeze()

	traces = raw_data['completed_trials_data']['traces_stim_aligned']#
	directions = raw_data['completed_trials_data']['stim_dir']

	independent_variable= np.array(directions)
	data_to_avg = np.array(traces)
	av_traces = []
	for x in np.unique(independent_variable):
		av_traces.append(np.mean(data_to_avg[np.where(independent_variable==x)[0],:,:],axis=0))
	av_traces = np.array(av_traces)

	return av_traces, data_to_avg, independent_variable, stim
import numpy as np
import scipy.io as sio
import glob
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import cycle
import pystan
from scipy.optimize import curve_fit
from scipy import stats
import time
import matplotlib.patches as mpatches
from matplotlib import cm

color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])



def load_RDK_data_from_mat_to_pandas(filename):
    mat = sio.loadmat(filename)
    mat = mat['temp_obj'][0]
    field_names = mat.dtype.names
    data = mat[0]
    data_dict = dict(zip(field_names,data))
    for key in data_dict:
        data_dict[key] = data_dict[key].squeeze()
    data = pd.DataFrame.from_dict(data_dict)
    data.index.name = 'trial'

    return data



def compile_graphical_physch_curve():
    simple_model = """
        data {
            int<lower=0> N;
            real c[N];
            int r[N];
        }

        parameters {
            real<lower=0.0,upper=1.0> y0;
            real<lower=0.0,upper=1.0> a;
            real<lower=-1.0,upper=1.0> c0;
            real<lower=0.0,upper=1.0> b;
        }

        transformed parameters {
            vector[N] theta;
            for (i in 1:N)
                theta[i] = y0+a/(1+exp(-1*(c[i]-c0)/b));
        }

        model {
                y0 ~ beta(2,10);
                a ~beta(10,2);
                c0 ~ uniform(-1,1);
                b ~ uniform(0,1);
                for (i in 1:N)
                    r[i] ~ bernoulli(theta[i]);
        } """

    print('compiling graphical model')
    sm =  pystan.StanModel(model_code=simple_model)
    return sm



def fit_psych_curve_fast(data):
    # fit least square regression model

    def func(c,y0,a,c0,b):
        return y0+a/(1+np.exp(-1*(c-c0)/b))

    x_data = data.coherence.values
    y_data = data.response_right.values

    popt, pcov = curve_fit(func,x_data,y_data, bounds = ((0.0,-1.0,-1.0,-np.inf),(1.0,1.0,1.0,np.inf)))
    return popt

def plot_sigmoid(p,color='k',alf=1.):
    x_val = np.linspace(-1,1,200).reshape(200,1)
    plt.plot(x_val,p[0]+p[1]/(1+np.exp(-1*(x_val-p[2])/p[3])),color,alpha=alf)



def analyze_day_fast(data):
    # fit day using nonlinear least squares

    print('fitting all day aggregated')
    fits_all = fit_psych_curve_fast(data)

    print('fitting by prior')
    fits_prior = []
    for p in data.prior_right.unique():
        print('fitting prior '+str(p))
        data_p = data.loc[lambda x: x.prior_right == p] # pick out data for the given prior
        fits_prior.append(fit_psych_curve_fast(data_p))


    print('fitting by block')
    fits_block = []
    block_length = 100
    num_blocks = int(np.rint(len(data)/block_length))
    for i in range(num_blocks):
        i_start = i*block_length
        i_end = i*block_length+block_length-1
        print('fitting block '+str(i))
        if i_end > len(data)-1:
            i_end = len(data)-1
        data_block = data.loc[i_start:i_end]
        fits_block.append(fit_psych_curve_fast(data_block))


    return fits_all, fits_prior, fits_block




def plot_fast_day_analysis(fast_analysis):

    plt_idx = 1
    plt.subplot(3,3,plt_idx)
    plot_sigmoid(fast_analysis[0])

    plt_idx = plt_idx+1
    for i in range(len(fast_analysis[1])):
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[1][i])

    plt_idx = plt_idx+1
    for i in range(len(fast_analysis[2])):
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[2][i])


    for i in range(len(fast_analysis[2])):
        plt_idx = plt_idx+1
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[2][i])

def plot_fast_day_analysis_data(fast_analysis,data):

    plt.figure(figsize=(20,20))
    colors = ['b','g','r','c','m']


    prior_list = data.prior_right.unique()
    plt_idx = 1
    plt.subplot(3,3,plt_idx)
    plot_sigmoid(fast_analysis[0])

    data_p = data # pick out data for the given prior
    coherence_means = data_p.groupby(['coherence'],as_index=False).response_right.mean()

    # std of estimation for a binary variable is sqrt(p(1-p)/n)
    yerrors = np.ravel(coherence_means.apply(lambda x: x*(1-x)).response_right)
    yerrors = np.sqrt(yerrors)/np.ravel(data_p['coherence'].value_counts())
    plt.errorbar(coherence_means.coherence,coherence_means.response_right,yerr = yerrors,fmt='.',color='b')
    plt.xlim([-1,1])
    plt.ylim([-0.1,1.1])
    plt_idx = plt_idx+1
    for i in range(len(fast_analysis[1])):
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[1][i])
        plt.plot([fast_analysis[1][i][2],fast_analysis[1][i][2]],[0,1])
        data_p = data.loc[lambda x: x.prior_right == prior_list[i]]# pick out data for the given prior
        coherence_means = data_p.groupby(['coherence'],as_index=False).response_right.mean()

        # std of estimation for a binary variable is sqrt(p(1-p)/n)
        yerrors = np.ravel(coherence_means.apply(lambda x: x*(1-x)).response_right)
        yerrors = np.sqrt(yerrors)/np.ravel(data_p['coherence'].value_counts())
        plt.errorbar(coherence_means.coherence,coherence_means.response_right,yerr = yerrors,fmt='.',color=colors[i])
        plt.xlim([-1,1])
        plt.ylim([0,1])
    plt_idx = plt_idx+1
    for i in range(len(fast_analysis[2])):
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[2][i])

        plt.xlim([-1,1])
        plt.ylim([-0.1,1.1])

    fits_block = []
    block_length = 100
    num_blocks = int(np.rint(len(data)/block_length))
    for i in range(num_blocks):
        i_start = i*block_length
        i_end = i*block_length+block_length-1
        print('fitting block '+str(i))
        if i_end > len(data)-1:
            i_end = len(data)-1
        data_block = data.loc[i_start:i_end]
        data_p = data_block
        coherence_means = data_p.groupby(['coherence'],as_index=False).response_right.mean()

        # std of estimation for a binary variable is sqrt(p(1-p)/n)
        yerrors = np.ravel(coherence_means.apply(lambda x: x*(1-x)).response_right)
        yerrors = np.sqrt(yerrors)/np.ravel(data_p['coherence'].value_counts())
        plt.errorbar(coherence_means.coherence,coherence_means.response_right,yerr = yerrors,fmt='.',color=colors[i])

        plt.xlim([-1,1])
        plt.ylim([-0.1,1.1])







    for i in range(len(fast_analysis[2])):
        plt_idx = plt_idx+1
        plt.subplot(3,3,plt_idx)
        plot_sigmoid(fast_analysis[2][i])

def fit_psych_curve(data, sm):
    # sampling and optimization of 4 parameter psych curve function

    data_dict = {'N': len(data.index),
               'c': data.coherence.values,
               'r': data.response_right.values}

    def init_params(chain_id=1):
        return {'y0':0.2,'a':0.7,'c0':0.0,'b':0.2}

    #print('starting optimization')
    #op = sm.optimizing(data = data_dict, verbose = True)

    print('starting sampling')
    #fit = sm.sampling(data=data_dict, iter=2000, chains=1,warmup=1000,init = init_params)
    #fit = pystan.stan(model_code=simple_model, data=data_dict, iter=2000, chains=1,warmup=1000,init = init_params)
    fit = sm.sampling(data=data_dict, iter=2000, chains=1,warmup=1000,init = init_params)

    return fit


def analyze_day(data, sm):
    #t0 = time.time()

    def make_data_dict(data):
        return {'N': len(data.index),'c': data.coherence.values,'r': data.response_right.values}

    def init_params(chain_id=1):
        return {'y0':0.2,'a':0.7,'c0':0.0,'b':0.2}

    #print('fitting all day aggregated')
    fits_all = sm.sampling(data=make_data_dict(data), iter=2000, chains=1,warmup=1000,init = init_params)

    #print('fitting by prior')
    fits_prior = []
    for p in data.prior_right.unique():
        #print('fitting prior '+str(p))
        data_p = data.loc[lambda x: x.prior_right == p] # pick out data for the given prior
        fits_prior.append(sm.sampling(data=make_data_dict(data_p), iter=2000, chains=1,warmup=1000,init = init_params))


    #print('fitting by block')
    fits_block = []
    block_length = 100
    num_blocks = int(np.rint(len(data)/block_length))
    for i in range(num_blocks):
        i_start = i*block_length
        i_end = i*block_length+block_length-1
        #print('fitting block '+str(i))
        if i_end > len(data)-1:
            i_end = len(data)-1
        data_block = data.loc[i_start:i_end]
        fits_block.append(sm.sampling(data=make_data_dict(data_block), iter=2000, chains=1,warmup=1000,init = init_params))

    #print('sampling has taken ' + str((time.time()-t0)/60) + ' minutes')
    return fits_all, fits_prior, fits_block

def plot_PPD_days(analysis,datas):
    differences = []
    x_val = np.linspace(-1,1,200)
    colors = ['r', 'g', 'b']
    prior_list = datas[0]['prior_right'].unique()
    prior_dict = {}
    for i in range(len(prior_list)):
        prior_dict[prior_list[i]] = colors[i];

    plt.figure(figsize=(40,20))
    for d in range(len(analysis)):

        day_analysis = analysis[d]
        prior_for_day = datas[d]['prior_right'].unique()
        colors_for_day = [prior_dict[key] for key in prior_for_day]

        for j in range(len(day_analysis[1])):
            prior_for_plot = prior_for_day[j];
            plt.subplot(3,8,d+1)
            c = colors_for_day[j]
            for i in range(1000):
                fit = day_analysis[1][j]
                y0_mean, a_mean, c0_mean,b_mean = fit['y0'][i],fit['a'][i],fit['c0'][i],fit['b'][i]
                plt.plot(x_val, y0_mean+a_mean/(1+np.exp(-1*(x_val-c0_mean)/b_mean)),color=c ,alpha=0.006,linewidth=3)



                plt.ylim(0,1)
                plt.xlim([-1,1])


            #data_p = datas[d].loc[lambda x: x.prior_right == prior_for_plot] # pick out data for the given prior
            #coherence_means = data_p.groupby(['coherence'],as_index=False).response_right.mean()

            # std of estimation for a binary variable is sqrt(p(1-p)/n)
            #yerrors = np.ravel(coherence_means.apply(lambda x: x*(1-x)).response_right)
            #yerrors = np.sqrt(yerrors)/np.ravel(data_p['coherence'].value_counts())
            #plt.errorbar(coherence_means.coherence,coherence_means.response_right,yerr = yerrors,fmt='.',color='k', ms = 10, elinewidth=2, capthick=2)
            # logistic regression to fit psychophysical curve
            # log_model = linear_model.LogisticRegression(C=1e20,max_iter=100,penalty='l2')
            # log_model.fit(X = data_p.coherence.reshape(data_p.coherence.count(),1), y = np.ravel(data_p.response_right))

            # plot values psych curves
            # x_val = np.linspace(-1,1,200).reshape(200,1)
            # plt.plot(x_val,log_model.predict_proba(x_val)[:,1],color=c)

        plt.subplot(3,8,d+9)
        patches = []
        to_diff = []
        for j in range(len(day_analysis[1])):
            c = colors_for_day[j]
            c0_mean = day_analysis[1][j]['c0']
            plt.hist(c0_mean,bins=np.linspace(-1,1,200),normed=1,histtype='stepfilled',facecolor=c,linewidth=0,alpha=0.4,label=str(prior_for_day[j]))
            plt.xlim([-1,1])
            plt.legend()


        plt.subplot(3,8,d+17)
        if prior_for_day[0]<prior_for_day[1]:
            differences.append(day_analysis[1][0]['c0'] - day_analysis[1][1]['c0'])
            plt.hist(day_analysis[1][0]['c0'] - day_analysis[1][1]['c0'],bins=np.linspace(-1,1,100),normed=1,histtype='stepfilled',facecolor='k',linewidth=0)
        else:
            differences.append(day_analysis[1][1]['c0'] - day_analysis[1][0]['c0'])
            plt.hist(day_analysis[1][1]['c0'] - day_analysis[1][0]['c0'],bins=np.linspace(-1,1,100),normed=1,histtype='stepfilled',facecolor='k',linewidth=0)

    return differences


def plot_psych_curve_different_history(data):
    fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(40,20))


    histories = [1,2,3,4,5,10,15,20]
    for i in range(len(histories)):
        ix = np.unravel_index(i, axes.shape)
        data.groupby(['coherence','stim_right_mean'+str(histories[i])]).mean().unstack().response_right.plot(colormap=cm.rainbow,style='.-',ax=axes[ix])






def get_d_prime(data, trace_stim_av,subject):
    d1 = trace_stim_av[:,data.index[data[subject]==1].tolist()]
    d0 = trace_stim_av[:,data.index[data[subject]==0].tolist()]

    return np.divide(np.abs(d0.mean(axis=1)-d1.mean(axis=1)),d0.var(axis=1)+d1.var(axis=1))


def do_spearmans_test(data,trace_stim_av,coherences):
    spearmanr2 = []
    spearmanr_p_val = []

    for n in range(trace_stim_av.shape[0]):
        coherence_activity = []
        for i in coherences:
            coherence_activity.append(trace_stim_av[n,data.index[data['coherence']==i].tolist()].mean())
        spearmanr2.append(stats.mstats.spearmanr(coherences, coherence_activity, use_ties=True)[0])
        try:
            spearmanr_p_val.append(stats.mstats.spearmanr(coherences, coherence_activity, use_ties=True)[1].compressed()[0])
        except:
            spearmanr_p_val.append(stats.mstats.spearmanr(coherences, coherence_activity, use_ties=True)[1])
    spearmanr2 = np.array(spearmanr2)
    spearmanr_p_val = np.array(spearmanr_p_val)
    return spearmanr2,spearmanr_p_val



def tuning_types(data,trace_stim_av,coherences):

    plt.figure(figsize=(10,10))
    spearman, spearman_p_val = do_spearmans_test(data,trace_stim_av,coherences)


    tuned_inds_increasing = np.where((spearman_p_val < 0.05) & (spearman > 0) )[0]
    tuned_inds_decreasing = np.where((spearman_p_val < 0.05) & (spearman < 0) )[0]

    plt.subplot(2,2,1)
    for n in tuned_inds_increasing:
        coherence_activity = []
        for i in coherences:
            coherence_activity.append(trace_stim_av[n,data.index[data['coherence']==i].tolist()].mean())
        plt.plot(coherences, coherence_activity); plt.xlabel('coherence'); plt.ylabel('mean activity'); plt.title('increasing ('+str(len(tuned_inds_increasing)) + ')')

    plt.subplot(2,2,2)
    for n in tuned_inds_decreasing:
        coherence_activity = []
        for i in coherences:
            coherence_activity.append(trace_stim_av[n,data.index[data['coherence']==i].tolist()].mean())
        plt.plot(coherences, coherence_activity); plt.xlabel('coherence');plt.title('decreasing ('+str(len(tuned_inds_decreasing))+')')

    spearman_left, spearman_p_val_left = do_spearmans_test(data,trace_stim_av,coherences[coherences<0])
    spearman_right, spearman_p_val_right = do_spearmans_test(data,trace_stim_av,coherences[coherences>0])

    tuned_inds_left_increasing = np.where((spearman_p_val_left < 0.05) & (spearman_left > 0) )[0]
    tuned_inds_left_decreasing = np.where((spearman_p_val_left < 0.05) & (spearman_left < 0) )[0]

    tuned_inds_right_increasing = np.where((spearman_p_val_right < 0.05) & (spearman_right > 0) )[0]
    tuned_inds_right_decreasing = np.where((spearman_p_val_right < 0.05) & (spearman_right < 0) )[0]

    # find left increasing and right decreasing
    humps = np.intersect1d(tuned_inds_left_increasing,tuned_inds_right_decreasing)
    valleys = np.intersect1d(tuned_inds_left_decreasing,tuned_inds_right_increasing)
    plt.subplot(2,2,3)
    for n in humps:
        coherence_activity = []
        for i in coherences:
            coherence_activity.append(trace_stim_av[n,data.index[data['coherence']==i].tolist()].mean())
        plt.plot(coherences, coherence_activity); plt.xlabel('coherence'); plt.ylabel('mean activity'); plt.title('peak ('+str(len(humps)) + ')')

    plt.subplot(2,2,4)
    for n in valleys:
        coherence_activity = []
        for i in coherences:
            coherence_activity.append(trace_stim_av[n,data.index[data['coherence']==i].tolist()].mean())
        plt.plot(coherences, coherence_activity); plt.xlabel('coherence');plt.title('valley ('+str(len(valleys))+')')



    return tuned_inds_increasing, tuned_inds_decreasing, humps, valleys



def get_exponential_history(data,tau):

    output = []
    d=np.array(data.stim_right.shift(1).tolist())
    l=len(data.stim_right.tolist())
    output = []
    for i in range(l):
        d[np.arange(i)]=[0]
        e = np.exp(-np.arange(l-i-1)/tau)
        e = np.insert(e,0,np.zeros(i+1))
        e = e/np.sum(e)
        output.append(np.sum(np.multiply(e,d)))
    return output

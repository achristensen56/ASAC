import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle
import numpy as np
import pandas as pd

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    


def read_file(file_name):
    '''
    right now this will only work for bruno, 
    need to remove stuff about opto trial for 
    other rats.
    '''
    
    with open(file_name, 'rb') as f:
        raw_data = pickle.load(f)
        
    coherence = raw_data['coherence'][:, np.newaxis]

    prior = raw_data['rightward_prior'][:, np.newaxis]
    stim_dir = raw_data['stim_dir']
    resp_dir = raw_data['response_side']
    resp_dir = np.array([(dir == 'right') for dir in resp_dir])[:, np.newaxis]
    
    stim_dir = np.array([(dir == 'right') for dir in stim_dir])
    
    dirs = np.ones([len(stim_dir), 1])
    dirs[stim_dir == False] = -1
    
    coherence = coherence*dirs

    com_trials = raw_data['was_completed'].squeeze().astype('bool')
    cor_trials = raw_data['was_correct'].squeeze().astype('bool')[:, np.newaxis]
 
    correct = cor_trials[com_trials]
    coherence = coherence[com_trials]
    prior = prior[com_trials]
    
    resp_dir = resp_dir[com_trials]

  
    return  coherence, correct, resp_dir, prior

def read_data_list(data):

    completed_data = data["completed_trials_data"]
    coherence = abs(completed_data['noise'][:, np.newaxis])
    prior = completed_data['prior'][:, np.newaxis]

    stim_dir = completed_data['stim_dir']
    resp_dir = completed_data['response_side']
    resp_dir = np.array([(np.logical_or(dir == 1, dir == 'right')) for dir in resp_dir])[:, np.newaxis]

    stim_dir = np.array([np.logical_or(dir == 90, dir == 'right') for dir in stim_dir])
    
    dirs = np.ones([len(stim_dir), 1])
    dirs[stim_dir == False] = -1
    
    coherence = coherence*dirs
    
    correct = completed_data['was_correct'].squeeze().astype('bool')[:, np.newaxis]

    return coherence, correct, resp_dir, prior

def compile_graphical_psych_curve():
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

    return fits_all, fits_prior


def plot_psych_curve(data_file_list):

    sm = compile_graphical_psych_curve()

    #currently concatenates all data together from listed files before running analysis

    coh, cor, resp_d, pri, opt = [], [], [], [], []
    for file in data_file_list:
        
        coherence,  cor_trials, resp_dir, prior = read_data_list(file)

        coh.append(coherence)

        cor.append(cor_trials)
        resp_d.append(resp_dir)
        pri.append(prior)
        
        
    coherence, correct, resp_dir, prior = np.row_stack(coh).squeeze(), np.row_stack(cor).squeeze(),np.row_stack(resp_d).squeeze(), np.row_stack(pri).squeeze()


    data_dict = {'coherence': coherence, 'response_right': resp_dir.astype(int), 'prior_right': prior}
    data_df = pd.DataFrame(data = data_dict)

    fits_all, fits_prior = analyze_day(data_df, sm)

    x_val = np.linspace(-1,1,200)


    colors = ['m', 'r']   
        
    plt.figure(figsize = (15, 10))    
    ax = plt.subplot(221)
    for j, fit in enumerate(fits_prior):
        c = colors[j]
        y_vals = np.array([fit['y0'][i]+fit['a'][i]/(1+np.exp(-1*(x_val-fit['c0'][i])/fit['b'][i])) for i in range(len(fit['y0']))])
        plt.plot(x_val, y_vals.mean(axis = 0), linewidth = 3, color = c)
        plt.fill_between(x_val, y_vals.mean(axis = 0) - y_vals.std(axis = 0), 
                         y_vals.mean(axis = 0) + y_vals.std(axis = 0), color = c, alpha = .1)
        
        
    for j, p in enumerate(sorted(data_df.prior_right.unique())):
        
        coherence_means = data_df[data_df.prior_right ==p].groupby(['coherence'],as_index=False).response_right.mean()
        # std of estimation for a binary variable is sqrt(p(1-p)/n)
        yerrors = np.ravel(coherence_means.apply(lambda x: x*(1-x)).response_right)
        yerrors = np.sqrt(yerrors)/np.ravel(data_df[data_df.prior_right ==p]['coherence'].value_counts())
        plt.errorbar(coherence_means.coherence,coherence_means.response_right,yerr = yerrors,fmt='.',color=colors[j], label = p)

    plt.ylim(0,1)
    plt.xlim([-1,1])
    simpleaxis(ax)
    plt.legend(frameon = False)
    plt.savefig("psych_curve.eps", dpi = 300)

    return fits_prior
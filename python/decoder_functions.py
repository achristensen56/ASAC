
import numpy as np
import scipy.io as sio
import glob
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import cycle
from scipy import ndimage
#import pystan
from scipy.optimize import curve_fit
from scipy import stats
import time
import matplotlib.patches as mpatches
from matplotlib import cm
import pyprind
color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])


#first let's try to decode left versus right stimulus, using only correct trials, concensus classifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC



def get_exponential_weighted_data(data_to_shift,tau):

    output = []
    d = data_to_shift; l = len(d)

    for t in range(l):
        sum = 0
        for k in range(t-1):
            sum = sum + data_to_shift[k] * np.exp(-(t-1-k)/tau)
        output.append(sum)

    return np.array(output)




def balanced_train_test_split(data,balance_category, vals = [0,1]):

    # amount of data to use for training, everything else will be testing
    ratio = 0.90


    # split into stim right and stim left data
    data_r = data[data[balance_category]==vals[1]]
    data_l = data[data[balance_category]==vals[0]]

    # find minimun sample size
    min_data_size = np.min([len(data_r),len(data_l)])

    # take random sample of the data
    data_r_ = data_r.sample(int(np.floor(min_data_size)),replace=False)
    data_l_ = data_l.sample(int(np.floor(min_data_size)),replace=False)


    # take first of the sample as training data and second half of the sample as testing data
    training_indices = data_r_.index.tolist()[:int(len(data_r_)*ratio)]+ \
                        data_l_.index.tolist()[:int(len(data_l_)*ratio)]
    testing_indices = data_r_.index.tolist()[int(len(data_r_)*ratio):]+\
                        data_l_.index.tolist()[int(len(data_l_)*ratio):]

    training_data = data.loc[training_indices]
    testing_data = data.loc[testing_indices]

    return training_data, testing_data, training_indices, testing_indices


def balanced_train_test_split_multi(data,balance_category):


    # amount of data to use for training, everything else will be testing
    ratio = 0.90


    labels = data[balance_category].unique()
    labels = labels[~np.isnan(labels)]
    # split into stim right and stim left data
    data_l = []
    for i in range(len(labels)):
        data_l.append(data[data[balance_category]==labels[i]])


    # find minimun sample size
    min_data_size = np.min([len(x) for x in data_l])

    # take random sample of the data
    data_l_ = []
    for i in range(len(labels)):
        data_l_.append(data_l[i].sample(int(np.floor(min_data_size)),replace=False))



    # take first of the sample as training data and second half of the sample as testing data
    training_indices = []
    for i in range(len(labels)):
        training_indices = training_indices + data_l_[i].index.tolist()[:int(len(data_l_[i])*ratio)]

    testing_indices = []
    for i in range(len(labels)):

        testing_indices = testing_indices + data_l_[i].index.tolist()[int(len(data_l_[i])*ratio):]

    training_data = data.loc[training_indices]
    testing_data = data.loc[testing_indices]

    return training_data, testing_data, training_indices, testing_indices




def test_model(model,testing_data,time,traces,category):

    # find the indices for the testing data
    indices = testing_data.index.tolist()

    # make predictions with the relevant data
    yhat = model.predict(traces[:,indices,time].T)


    # return the accuracy
    return np.sum((yhat == testing_data.loc[indices][category])/len(yhat))


def test_model_linear(model,testing_data,time,traces,category,shuffle=False):

    # find the indices for the testing data
    indices = testing_data.index.tolist()

    # make predictions with the relevant data
    yhat = model.predict(traces[:,indices,time].T)

    if shuffle == True:
        np.random.shuffle(indices)

        # return the accuracy
    return np.mean(np.sqrt((yhat - testing_data.loc[indices][category])**2))
    #return (np.array(yhat), np.array(testing_data.loc[indices][category].tolist()))


def train_decoder(traces, decoding_category,data,  decoder_type = 'logistic'):
    # for each timepoint, we have an array for each replicate, each of which has a dictionary of
    # (1) training data
    # (2) testing data
    # (3) the model
    # (4) accuracy on the testing set

    bar = pyprind.ProgBar(traces.shape[2],monitor=True,bar_char='█')
    # an array for each timepoint
    stimulus_decoder_results = []

    if decoder_type == 'logistic':
        for trial_time in range(traces.shape[2]):

            '''
            # get regularization
            reg_accs = []
            reg_vals = np.logspace(-2, 2, num=20)
            for reg_val in reg_vals:
                reg_accs_ = []
                for r in range(20):
                    training_data,testing_data,training_indices,testing_indices = \
                                        balanced_train_test_split_multi(data,decoding_category)

                    model = LogisticRegression(C = reg_val, penalty='l2')


                    model.fit(traces[:,training_indices,trial_time].T, training_data[decoding_category])


                    acc_ = test_model(model,testing_data,trial_time,traces,decoding_category)

                    # make a dictionary with the relevant data
                    reg_accs_.append(acc_)
                reg_accs.append(np.mean(reg_accs_))
            print(reg_accs)
              '''


            models_ = []

            for repeats in range(50):

                # seperate data, balancing by leftward and rightward trials, using only correct trials
                training_data,testing_data,training_indices,testing_indices = \
                                    balanced_train_test_split_multi(data,decoding_category)

                # set up a logistic regression problem
                model = LogisticRegression(C = 1, penalty='l2')

                # fit the model
                model.fit(traces[:,training_indices,trial_time].T, training_data[decoding_category])

                # get the accuracy on the testing set
                acc_ = test_model(model,testing_data,trial_time,traces,decoding_category)

                # make a dictionary with the relevant data
                models_.append({'training_data':training_data,'testing_data':testing_data,'model':model,'accuracy':acc_})

            # put the array of dictionaries into the reults array
            stimulus_decoder_results.append(models_)

            bar.update()
    elif decoder_type == 'linear':

        for trial_time in range(traces.shape[2]):

                models_ = []

                for repeats in range(50):

                    # seperate data, balancing by leftward and rightward trials, using only correct trials
                    training_data,testing_data,training_indices,testing_indices = \
                                        balanced_train_test_split_linear(data,decoding_category)

                    # set up a logistic regression problem
                    model = LinearRegression()
                    #model = HuberRegressor()
                    # fit the model
                    model.fit(traces[:,training_indices,trial_time].T, training_data[decoding_category])

                    # get the accuracy on the testing set
                    acc_ = test_model_linear(model,testing_data,trial_time,traces,decoding_category)
                    shuf_acc_ = test_model_linear(model,testing_data,trial_time,traces,decoding_category,shuffle=True)

                    # make a dictionary with the relevant data
                    models_.append({'training_data':training_data,'testing_data':testing_data,'model':model,'accuracy':acc_,'shuffled_accuracy':shuf_acc_})

                # put the array of dictionaries into the reults array
                stimulus_decoder_results.append(models_)

                bar.update()

    return stimulus_decoder_results



def get_accuracy_over_trial(decoder_results):
    acc_vec = []

    for time,timepoint_array in enumerate(decoder_results):
        acc_ = []
        for repeat in timepoint_array:
            acc_.append(repeat['accuracy'])
        acc_vec.append(np.mean(acc_))

    to_plot = 100*ndimage.gaussian_filter1d(np.array(acc_vec), sigma = .5)
    return to_plot



def train_behavior_decoder(data,decoding_category,regressor_list):

    # build regressor_matrix (n_samples,n_features)


    models_ = []

    for repeats in range(50):

        # seperate data, balancing by leftward and rightward trials, using only correct trials
        training_data,testing_data,training_indices,testing_indices = \
                                    balanced_train_test_split_multi(data,decoding_category)



        # set up a logistic regression problem
        model = LogisticRegression(C = 100, penalty='l2')

        # fit the model
        model.fit(training_data[regressor_list].as_matrix(), training_data[decoding_category].as_matrix())

        # get the accuracy on the testing set
        acc_ = test_behavior_model(model,testing_data,regressor_list,decoding_category)

                # make a dictionary with the relevant data

        models_.append({'training_data':training_data,'testing_data':testing_data,'model':model,'accuracy':acc_})

    return models_


def test_behavior_model(model,testing_data,regressor_list,decoding_category):

    # find the indices for the testing data
    indices = testing_data.index.tolist()

    # make predictions with the relevant data
    yhat = model.predict(testing_data[regressor_list].as_matrix())

    # return the accuracy
    return np.sum((yhat == testing_data[decoding_category])/len(yhat))



def get_decoder_results_weights(decoder_results):
    acc_vec = []

    for time,timepoint_array in enumerate(decoder_results):
        acc_ = []
        for repeat in timepoint_array:
            acc_.append(repeat['model'].coef_)
        #print(np.squeeze(np.array(acc_)).shape)
        acc_vec.append(np.mean(acc_,axis=0))

    #to_plot = 100*ndimage.gaussian_filter1d(np.array(acc_vec), sigma = 2.5)
    return np.squeeze(np.array(acc_vec))



def balanced_train_test_split_linear(data,balance_category):

    # amount of data to use for training, everything else will be testing
    ratio = 0.75

    # split into stim right and stim left data

    data_r = data[data[balance_category]<0.5]
    data_l = data[data[balance_category]>0.5]


    #data_r = data[data['stim_right']==1]
    #data_l = data[data['stim_right']==0]

    # find minimun sample size
    min_data_size = np.min([len(data_r),len(data_l)])



    # take random sample of the data
    data_r_ = data_r.sample(int(np.floor(min_data_size)),replace=False)
    data_l_ = data_l.sample(int(np.floor(min_data_size)),replace=False)


    # take first of the sample as training data and second half of the sample as testing data
    training_indices = data_r_.index.tolist()[:int(len(data_r_)*ratio)]+ \
                        data_l_.index.tolist()[:int(len(data_l_)*ratio)]
    testing_indices = data_r_.index.tolist()[int(len(data_r_)*ratio):]+\
                        data_l_.index.tolist()[int(len(data_l_)*ratio):]


    training_data = data.loc[training_indices].dropna(subset=[balance_category])
    testing_data = data.loc[testing_indices].dropna(subset=[balance_category])




    return training_data, testing_data, training_data.index.tolist(), testing_data.index.tolist()




from scipy.ndimage import gaussian_filter

def pick_regularization(X, y, n_reg_points = 20):

    reg_vals = np.logspace(-2, 2, num=n_reg_points)
    #reg_vals = np.sort(reg_vals)

    #print(reg_vals)

    reg_arr = []
    reg_std = []

    for c in reg_vals:
        temp = []

        for i in range(20):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model = LogisticRegression(C = c, penalty= 'l2')
            model.fit(X_train, y_train)
            temp.append(model.score(X_test, y_test))


    reg_arr.append(np.mean(temp))
    reg_std.append(np.std(temp))


    reg_arr = np.array(reg_arr)
    reg_std = np.array(reg_std)

    reg_arr = gaussian_filter(reg_arr, sigma = 2)
    val = np.argmax(reg_arr)

    reg_val = reg_vals[val]

    #print("Selected regularization value {0}".format(reg_val))
    return reg_val

    #plt.semilogx(reg_vals, reg_arr)
    #plt.fill_between(reg_vals, reg_arr - reg_std, reg_arr + reg_std, alpha = .1)

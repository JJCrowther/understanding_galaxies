# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# scipy for algorithms
import scipy
from scipy import stats

# pymc3 for Bayesian Inference, pymc built on t
import pymc3 as pm
import theano.tensor as tt
import scipy
from scipy import optimize

# matplotlib for plotting
import matplotlib.pyplot as plt
#matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf

print('\nstart')

# Logistic function with both beta and alpha
def logistic(x, beta, alpha=0, gamma=1):
    return gamma / (1.0 + np.exp(np.dot(beta, x) + alpha))

if __name__ == '__main__':

    N_SAMPLES = 5000
    rounding=0.02
    scale_factor_data={}
    cut_threshold = 0.7
    full_data_array_first_cut=np.zeros((0, 5))

        # The data

    file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

    i=0 #index used for scale facotr multiplication
    parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift', 'elpetro_absmag_r'])

    for file_name in file_name_list:
        i+=1 #will cause problems if scale factors aren't linearly listed in intervals of 1, should change

        scale_factor_data[file_name] = frf.file_reader(file_name)

        scale_factor_dataframe = pd.DataFrame(scale_factor_data[file_name])
        scale_factor_dataframe.rename(columns={0: 'iauname', 1:'smooth-or-featured_smooth_pred', 2:'smooth-or-featured_featured-or-disk_pred', 3:'smooth-or-featured_artifact_pred'}, inplace=True) #rename the headers of the dataframe

        scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(i), '', regex=False)
        scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('.png', '', regex=False)

        merged_dataframe = scale_factor_dataframe.merge(parquet_file, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
        merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors
        merged_dataframe['redshift']=merged_dataframe['redshift'].mul(i) #Multiplies the redshift by the scalefactor

        first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -20)]
        #second_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -20 ) & (merged_dataframe["elpetro_absmag_r"] >= -21)]
        #third_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -21 ) & (merged_dataframe["elpetro_absmag_r"] >= -24)]

        merged_numpy_first_cut = first_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
        #merged_numpy_second_cut = second_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
        #merged_numpy_third_cut = third_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation

        numpy_merged_probs_first_cut = frf.prob_maker(merged_numpy_first_cut)
        #numpy_merged_probs_second_cut = frf.prob_maker(merged_numpy_second_cut)
        #numpy_merged_probs_third_cut = frf.prob_maker(merged_numpy_third_cut)

        full_data_array_first_cut=np.vstack((full_data_array_first_cut, numpy_merged_probs_first_cut)) #stacks all data from current redshift to cumulative array
        #full_data_array_second_cut=np.vstack((full_data_array_second_cut, numpy_merged_probs_second_cut)) #stacks all data from current redshift to cumulative array
        #full_data_array_third_cut=np.vstack((full_data_array_third_cut, numpy_merged_probs_third_cut)) #stacks all data from current redshift to cumulative array

        full_dataframe = pd.DataFrame(full_data_array_first_cut[:, 1:5].astype(float))

        full_dataframe.loc[full_dataframe[0] < cut_threshold, 0] = 0
        full_dataframe.loc[full_dataframe[0] > cut_threshold, 0] = 1

        redshift = np.array(full_dataframe.loc[:, 3])
        smooth = np.array(full_dataframe.loc[:, 0])
        # The modeling 

    with pm.Model() as sleep_model:
        # Create the alpha and beta parameters
        alpha = pm.Normal('alpha', mu=0.0, tau=0.01, testval=-2.0)
        beta = pm.Normal('beta', mu=0.0, tau=0.01, testval=2.0)
        gamma = pm.Normal('gamma', mu=0.0, tau=0.01, testval=0.7)
        
        # Create the probability from the logistic function
        p = pm.Deterministic('p', gamma / (1. + tt.exp(beta * redshift + alpha)))
        
        # Create the bernoulli parameter which uses the observed dat
        observed = pm.Bernoulli('obs', p, observed=smooth)
        
        # Starting values are found through Maximum A Posterior estimation
        # start = pm.find_MAP()
        
        # Using Metropolis Hastings Sampling
        step = pm.Metropolis()
        
        # Sample from the posterior using the sampling method
        #smooth_trace = pm.sample(N_SAMPLES, step=step, njobs=2);
        smooth_trace = pm.sample(N_SAMPLES, step=step);


        # Extract the alpha and beta samples
    alpha_samples = smooth_trace["alpha"][5000:, None]
    beta_samples = smooth_trace["beta"][5000:, None]
    gamma_samples = smooth_trace["gamma"][5000:, None]

    
        #plot the histogram of samples  
    #figsize(16, 10)

    #plt.subplot(311)
    #plt.title(r"""Distribution of $\alpha$ with %d samples""" % N_SAMPLES)

    #plt.hist(alpha_samples, histtype='stepfilled', 
    #        color = 'darkred', bins=30, alpha=0.8, density=True);
    #plt.ylabel('Probability Density')


    #plt.subplot(312)
    #plt.title(r"""Distribution of $\beta$ with %d samples""" % N_SAMPLES)
    #plt.hist(beta_samples, histtype='stepfilled', 
    #        color = 'darkblue', bins=30, alpha=0.8, density=True)
    #plt.ylabel('Probability Density')


    #plt.subplot(313)
    #plt.title(r"""Distribution of $\gamma$ with %d samples""" % N_SAMPLES)
    #plt.hist(gamma_samples, histtype='stepfilled', 
    #        color = 'darkgreen', bins=30, alpha=0.8, density=True)
    #plt.ylabel('Probability Density');

    #plt.show()
    

    # Time values for probability prediction
    redshifting_est = np.linspace(redshift.min(), redshift.max(), 1000)[:, None]

    # Take most likely parameters to be mean values
    alpha_est = alpha_samples.mean()
    beta_est = beta_samples.mean()
    gamma_est = gamma_samples.mean()

    # Probability at each time using mean values of alpha and beta
    smooth_est = logistic(redshifting_est, beta=beta_est, alpha=alpha_est, gamma=gamma_est)

    #Plotting smooth_est
    figsize(16, 6)

    plt.plot(redshifting_est, smooth_est, color = 'navy', 
            lw=3, label="Most Likely Logistic Model")
    plt.title('Probability Distribution for Sleep with %d Samples' % N_SAMPLES);
    plt.legend(prop={'size':18})
    plt.ylabel('Probability')
    plt.xlabel('PM Redshift');
    
    plt.show()
print('\nfin')
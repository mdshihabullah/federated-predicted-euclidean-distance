import random
import sys
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
import participant_utils as pu
import coordinator_utils as cu

# Declare for experimental purpose
total_no_samples=5000
D1_sample_size = random.randint(1, total_no_samples)
D2_sample_size = total_no_samples - D1_sample_size
dimension=20000

# Create the dataset accordingly
# D1, true_label_D1, spikes_D1 = make_blobs(n_samples=D1_sample_size,
#                                           n_features=dimension, 
                                          
#                                           cluster_std=random.uniform(0, 1),
#                                           return_centers=True)
D1 = np.random.random((D1_sample_size, dimension)) * 2 - 1
# D2, true_label_D2, spikes_D2 = make_blobs(n_samples=D2_sample_size,
#                                           n_features=dimension, 
                                          
#                                           cluster_std=random.uniform(0, 1),
#                                           return_centers=True)

D2 = np.random.random((D2_sample_size, dimension)) * 2 - 1
#### Participant Based Computation ####

# Each participant generates random spike in points 
# which in production environment will be shared to coordinator for creating overall spike array
generated_spikes_D1 = pu.generate_n_spikes_using_variance(0.90, D1, int(sys.argv[1]))
generated_spikes_D2 = pu.generate_n_spikes_using_variance(0.90, D2, int(sys.argv[1]))

#### Coordinator Based Computation ####
generated_spikes = np.concatenate((generated_spikes_D1, generated_spikes_D2))
print("Shape of Generated Spikes",generated_spikes.shape)

# pu.plot3dwithspike(width=9, height=6, title= "Dataset with spike points", datapoints = clustered_dataset, spikes=generated_spikes, myLabel=true_label)
# # rows are s1,s2..sn while columns are datapoints
euc_dist_D1_spikes = euclidean_distances(D1,generated_spikes)
# print("Spike local distance matrix of 1st participant: \n", euc_dist_D1_spikes)

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D2_spikes = euclidean_distances(D2,generated_spikes)
# print("Spike local distance matrix of 2nd participant: \n", euc_dist_D2_spikes)
  
slope_intercept_D1 = pu.regression_per_client(data= D1,
                                           euc_dist_data_spike= euc_dist_D1_spikes,
                                           regressor="Linear")
slope_intercept_D2 = pu.regression_per_client(data= D2,
                                           euc_dist_data_spike= euc_dist_D2_spikes,
                                           regressor="Huber")

global_true_euc_dist = euclidean_distances(np.concatenate((D1, D2)))

global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
cu.spearman_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)

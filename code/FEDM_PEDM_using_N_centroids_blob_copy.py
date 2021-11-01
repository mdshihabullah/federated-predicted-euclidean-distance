import random
import sys
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
import participant_utils as pu
import coordinator_utils as cu


# Declare for experimental purpose
total_no_samples = 100
dimension = 1200
# D1_sample_size = random.randint(1, total_no_samples)
# D2_sample_size = total_no_samples - D1_sample_size
# Read the dataset accordingly
clustered_dataset, true_label, centroids = make_blobs(n_samples=total_no_samples,
                                                      n_features=dimension, 
                                                      centers=int(sys.argv[1])*2,
                                                      cluster_std=random.uniform(0, 1),
                                                      return_centers=True)

#For simulating uneven distribution
np.random.shuffle(clustered_dataset)
D1,D2,D3,D4 = np.array_split(clustered_dataset, 4)
D2=np.concatenate((D2,D3,D4))
#### Participant Based Computation ####
# centroids_D1 = KMeans(n_clusters= int(sys.argv[1])).fit(D1).cluster_centers_
# centroids_D2 =KMeans(n_clusters= int(sys.argv[1])).fit(D2).cluster_centers_

#### Coordinator Based Computation ####
generated_spikes = centroids
print("Shape of Generated Spikes",generated_spikes.shape)

clustered_dataset = np.concatenate((D1, D2))
clustered_dataset_3d = cu.perform_PCA(3, clustered_dataset)
generated_spikes_3d = cu.perform_PCA(3, generated_spikes)
pu.plot3dwithspike(width=9, height=6, title= "Dataset with spike points", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=true_label)
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

global_true_euc_dist = euclidean_distances(clustered_dataset)

global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
cu.spearman_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
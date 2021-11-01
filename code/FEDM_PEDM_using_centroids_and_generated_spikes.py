from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import participant_utils as pu
import coordinator_utils as cu
from sklearn.cluster import AffinityPropagation


# Read the dataset accordingly
path = Path(__file__).parent / "../dataset/GSE84433_series_matrix.txt"
clustered_dataset = pd.read_csv(path, comment='!', sep="\t", header=0)
clustered_dataset = clustered_dataset.T
clustered_dataset.dropna(inplace=True)
clustered_dataset, clustered_dataset.columns = clustered_dataset[1:] , clustered_dataset.iloc[0]

# Convert the dataset into numpy ndarray for further computation
clustered_dataset = clustered_dataset.to_numpy(dtype='float64')

#For simulating equal distribution
# D1, D2 = np.array_split(clustered_dataset, 2)

#For simulating uneven distribution
np.random.shuffle(clustered_dataset)
D1,D2,D3,D4 = np.array_split(clustered_dataset, 4)
D2=np.concatenate((D2,D3,D4))

#### Participant Based Computation ####
centroids_D1 = AffinityPropagation().fit(D1).cluster_centers_
generated_spikes_D1 = pu.generate_spikes_with_centroids(D1, centroids_D1)
centroids_D2 = AffinityPropagation().fit(D2).cluster_centers_
generated_spikes_D2 = pu.generate_spikes_with_centroids(D2, centroids_D2)


#### Coordinator Based Computation ####
generated_spikes = np.concatenate((generated_spikes_D1, generated_spikes_D2))
print("Shape of Generated Spikes",generated_spikes.shape)
clustered_dataset_3d = cu.perform_PCA(3, clustered_dataset)
generated_spikes_3d = cu.perform_PCA(3, generated_spikes)
pu.plot3dwithspike(width=9, height=6, title= "Dataset with spike points", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=None)
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
                                           regressor="Linear")

global_true_euc_dist = euclidean_distances(clustered_dataset)

global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
cu.spearman_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
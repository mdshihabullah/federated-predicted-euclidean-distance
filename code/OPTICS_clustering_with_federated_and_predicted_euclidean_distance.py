import participant_utils as pu
import coordinator_utils as cu
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt


# Declare for experimental purpose
dimension=5000
total_no_samples=500
no_of_cluster = 5

# Generated dataset and use centroids as spike-in points
clustered_dataset, true_label, centroids = cu.generate_clustered_dataset(dimension=dimension,
                                                                   total_no_samples=total_no_samples,
                                                                   no_of_cluster= no_of_cluster, random_state=1)

# Plot the dataset and spike-in points for visualization
pu.plot3dwithspike(width=9, height=6, title= "Clustering with actual labels", datapoints = clustered_dataset, spikes=centroids, myLabel=true_label)

# Spliting the aggregated dataset for two participants
D1,D2 = np.array_split(clustered_dataset, 2)

# Each participant generates random spike in points 
# which in production environment will be shared to coordinator for creating overall spike array
generated_spikes_D1 = pu.generate_spikes_each_participant(D1)
generated_spikes_D2 = pu.generate_spikes_each_participant(D2)

# Concatenate the locally generated spikes to create global spike array
generated_spikes = np.concatenate((generated_spikes_D1, generated_spikes_D2))

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D1_spikes = euclidean_distances(D1,generated_spikes)
# print("Spike local distance matrix of 1st participant: \n", euc_dist_D1_spikes)

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D2_spikes = euclidean_distances(D2,generated_spikes)
# print("Spike local distance matrix of 2nd participant: \n", euc_dist_D2_spikes)

# Calculate and get slope and intercept for 1st participant's dataset  
slope_intercept_D1 = pu.regression_per_client(data= D1,
                                           euc_dist_data_spike= euc_dist_D1_spikes,
                                           regressor="Huber")

# Calculate and get slope and intercept for 2nd participant's dataset  
slope_intercept_D2 = pu.regression_per_client(data= D2,
                                           euc_dist_data_spike= euc_dist_D2_spikes,
                                           regressor="Linear")

global_true_euc_dist = euclidean_distances(clustered_dataset)

clustered_dataset_2d = cu.perform_PCA(2, clustered_dataset)
generated_spikes_2d = cu.perform_PCA(2, generated_spikes)

clustered_dataset_3d = cu.perform_PCA(3, clustered_dataset)
generated_spikes_3d = cu.perform_PCA(3, generated_spikes)

# https://stackoverflow.com/questions/59765712/optics-parallelism
label = OPTICS(metric='precomputed', n_jobs=-1).fit_predict(global_true_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gtdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with true distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(generated_spikes_2d[:,0] , generated_spikes_2d[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

pu.plot3dwithspike(width=9, height=6, title= "Clustering with true aggregated distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_gtdm)

cu.evaluation_scores(global_true_euc_dist, "Aggregated True Distance Matrix",  true_label, pred_label_gtdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)

global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

label = OPTICS(metric='precomputed', n_jobs=-1).fit_predict(global_fed_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gfdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with federated distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(generated_spikes_2d[:,0] , generated_spikes_2d[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

pu.plot3dwithspike(width=9, height=6, title= "Clustering with globally federated distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_gfdm)

cu.evaluation_scores(global_fed_euc_dist, "Global Federated Distance Matrix",  pred_label_gtdm, pred_label_gfdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)


label = OPTICS(metric='precomputed', n_jobs=-1).fit_predict(global_pred_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_2 =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(generated_spikes_2d[:,0] , generated_spikes_2d[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

pu.plot3dwithspike(width=9, height=6, title= "Clustering with globally predicted distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_2)

cu.evaluation_scores(global_pred_euc_dist, "Global Predicted Distance Matrix",  pred_label_gtdm, pred_label_2, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)

cu.plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
cu.plotDistanceMatrix(global_true_euc_dist, title="True Global Distance Matrix")
cu.plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import participant_utils as pu
import coordinator_utils as cu
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Read the dataset accordingly
path = Path(__file__).parent / "../dataset/GSE84426_series_matrix.txt"
clustered_dataset = pd.read_csv(path, comment='!', sep="\t", header=0)
clustered_dataset = clustered_dataset.T
clustered_dataset.dropna(inplace=True)
clustered_dataset, clustered_dataset.columns = clustered_dataset[1:] , clustered_dataset.iloc[0]

#Without using sklearn's LabelEncoder()
# true_label = clustered_dataset.iloc[:,0].astype('category').cat.codes

label_encoder = LabelEncoder()
true_label = label_encoder.fit_transform(clustered_dataset.iloc[:,0])
# clustered_dataset = clustered_dataset.drop(columns="Gene_ID")

# Convert the dataset into numpy ndarray for further computation
clustered_dataset = clustered_dataset.to_numpy(dtype='float64')

#For simulating equal distribution
# D1, D2 = np.array_split(clustered_dataset, 2)

#For simulating uneven distribution
np.random.shuffle(clustered_dataset)
D1,D2,D3,D4 = np.array_split(clustered_dataset, 4)
D2=np.concatenate((D2,D3,D4))

#### Participant Based Computation ####

# Each participant generates random spike in points 
# which in production environment will be shared to coordinator for creating overall spike array
generated_spikes_D1 = pu.generate_spikes_using_PCA_and_variance(D1, induce=3)
generated_spikes_D2 = pu.generate_spikes_using_PCA_and_variance(D1, induce=3)

#### Coordinator Based Computation ####
generated_spikes = np.concatenate((generated_spikes_D1, generated_spikes_D2))
print("Shape of Generated Spikes",generated_spikes.shape)

# pu.plot3dwithspike(width=9, height=6, title= "Clustering with actual labels", datapoints = clustered_dataset, spikes=generated_spikes, myLabel=true_label)
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

#### Coordinator Based Computation ####
# FOR EVALUATION PURPOSE ONLY
global_true_euc_dist = euclidean_distances(clustered_dataset)

# Transform the dataset and spike points into 2D and 3D for visualization purpose
# pca,clustered_dataset_2d = pu.perform_PCA(3, clustered_dataset)
# variance_percentage = str(float(np.round(pca.explained_variance_ratio_.cumsum()[2]*100, 1)))
# plot_title = "GSE84426 dataset with 3 Principal Components covering " + variance_percentage+"% variance"
clustered_dataset_2d = cu.perform_PCA(2, clustered_dataset)
generated_spikes_2d = cu.perform_PCA(2, generated_spikes)
# generated_spikes_3d = cu.perform_PCA(3, generated_spikes)
# plt.title(plot_title, fontsize='medium', pad=20)
# ax = plt.axes(projection='3d')
# ax.scatter3D(clustered_dataset_2d[:,0] , clustered_dataset_2d[:,1], clustered_dataset_2d[:,2])
# plt.show()

# pca,clustered_dataset_2d = pu.perform_PCA(2, clustered_dataset)
# variance_percentage = str(float(np.round(pca.explained_variance_ratio_.cumsum()[1]*100, 1)))
# plot_title = "GSE84426 dataset with 2 Principal Component covering " + variance_percentage+"% variance"
# clustered_dataset_3d = cu.perform_PCA(3, clustered_dataset)
# generated_spikes_2d = cu.perform_PCA(2, generated_spikes)
# generated_spikes_3d = cu.perform_PCA(3, generated_spikes)
# plt.title(plot_title, fontsize='medium', pad=20)
# plt.scatter(clustered_dataset_2d[:,0] , clustered_dataset_2d[:,1])
# plt.show()

# https://stackoverflow.com/questions/59765712/optics-parallelism
label = KMedoids(n_clusters=14, metric='precomputed',method='pam').fit_predict(global_true_euc_dist)

#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gtdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with true aggregated distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(generated_spikes_2d[:,0] , generated_spikes_2d[:,1] , s = 80, color = 'k')
plt.legend()
plt.savefig("GSE84426_K-medoids_ADM_2d.png")
# pu.plot3dwithspike(width=9, height=6, title= "Clustering with true aggregated distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_gtdm)

# cu.unsupervised_evaluation_scores(global_true_euc_dist, "Aggregated True Distance Matrix",  true_label, pred_label_gtdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)


global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

label = KMedoids(n_clusters=14, metric='precomputed',method='pam').fit_predict(global_fed_euc_dist)
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
plt.savefig("GSE84426_K-medoids_FEDM_2d.png")
# pu.plot3dwithspike(width=9, height=6, title= "Clustering with globally federated distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_gfdm)

cu.unsupervised_evaluation_scores(global_fed_euc_dist, "Global Federated Distance Matrix",  pred_label_gtdm, pred_label_gfdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)


MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

label = KMedoids(n_clusters=14, metric='precomputed',method='pam').fit_predict(global_pred_euc_dist)
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
plt.savefig("GSE84426_K-medoids_PEDM_2d.png")
# pu.plot3dwithspike(width=9, height=6, title= "Clustering with globally predicted distance matrix", datapoints = clustered_dataset_3d, spikes=generated_spikes_3d, myLabel=pred_label_2)

cu.unsupervised_evaluation_scores(global_pred_euc_dist, "Global Predicted Distance Matrix",  pred_label_gtdm, pred_label_2, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True)

cu.plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
cu.plotDistanceMatrix(global_true_euc_dist, title="True Global Distance Matrix")
cu.plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
cu.spearman_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)
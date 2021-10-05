from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import participant_utils as pu
import coordinator_utils as cu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the dataset accordingly
path = Path(__file__).parent / "../dataset/formattedGeneExpressionSmall.txt"
clustered_dataset = pd.read_csv(path, sep="\t", header=0)
# Drop rows with null values as they are around 6% of the dataset
clustered_dataset.dropna(inplace=True)
clustered_dataset = clustered_dataset.iloc[1: , :]
unique_label = clustered_dataset.Gene_ID.unique()
#Without using sklearn's LabelEncoder()
# true_label = clustered_dataset.Gene_ID.astype('category').cat.codes

label_encoder = LabelEncoder()
true_label = label_encoder.fit_transform(clustered_dataset.Gene_ID)
no_of_cluster = len(unique_label)
# Get spike points as the mean of of all the points in each label
spikes = pu.get_spikes_from_centroid(clustered_dataset, "Gene_ID", unique_label)
clustered_dataset = clustered_dataset.drop(columns="Gene_ID")
# Convert the dataset into numpy ndarray for further computation
clustered_dataset = clustered_dataset.to_numpy(dtype='float64')

#For simulating equal distribution
# D1, D2 = np.array_split(clustered_dataset, 2)

#For simulating uneven distribution
np.random.shuffle(clustered_dataset)
D1,D2,D3,D4 = np.array_split(clustered_dataset, 4)
D2=np.concatenate((D2,D3,D4))

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D1_spikes = euclidean_distances(D1,spikes)
# print("Spike local distance matrix of 1st participant: \n", euc_dist_D1_spikes)

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D2_spikes = euclidean_distances(D2,spikes)
# print("Spike local distance matrix of 2nd participant: \n", euc_dist_D2_spikes)
  
slope_intercept_D1 = pu.regression_per_client(data= D1,
                                           euc_dist_data_spike= euc_dist_D1_spikes,
                                           regressor="Huber")
slope_intercept_D2 = pu.regression_per_client(data= D2,
                                           euc_dist_data_spike= euc_dist_D2_spikes,
                                           regressor="Linear")

global_true_euc_dist = euclidean_distances(clustered_dataset)
clustered_dataset_2d = cu.perform_MDS(2, clustered_dataset)
label = KMeans(n_clusters=no_of_cluster, random_state=0).fit_predict(global_true_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gtdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with true distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

cu.unsupervised_evaluation_scores(global_true_euc_dist, "Aggregated True Distance Matrix",  true_label, pred_label_gtdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=True, davies_bouldin=False)

global_fed_euc_dist = cu.calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

label = KMeans(n_clusters=no_of_cluster, random_state=0).fit_predict(global_fed_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gfdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

cu.unsupervised_evaluation_scores(global_fed_euc_dist, "Global Federated Distance Matrix",  pred_label_gtdm, pred_label_gfdm, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=True, davies_bouldin=False)

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

label = KMeans(n_clusters=no_of_cluster, random_state=0).fit_predict(global_pred_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_2 =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

cu.unsupervised_evaluation_scores(global_pred_euc_dist, "Global Predicted Distance Matrix",  pred_label_gtdm, pred_label_2, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=True, davies_bouldin=False)

cu.plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
cu.plotDistanceMatrix(global_true_euc_dist, title="True Global Distance Matrix")
cu.plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)




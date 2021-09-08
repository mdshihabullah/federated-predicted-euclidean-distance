# Some of the imports will not be necessary in production
import math
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import seaborn as sns

def generate_clustered_dataset(dimension,total_no_samples,no_of_cluster, random_state):
    clustered_dataset, true_label, centroids = make_blobs(n_samples=total_no_samples,
                                                          n_features=dimension, 
                                                          centers=no_of_cluster,
                                                          return_centers=True, 
                                                          random_state=random_state) #check randomstate =80
    return clustered_dataset, true_label, centroids


def plot3dwithspike(width, height, datapoints, spikes, myLabel=None) :
    plt.figure(figsize=(width,height))
    plt.title("Clusters with random points", fontsize='medium')
    ax = plt.axes(projection='3d')
    ax.scatter3D(datapoints[:, 0], datapoints[:,1], datapoints[:,2], c=myLabel, marker='o',  s=15, edgecolor='k')
    ax.scatter3D(spikes[:, 0], spikes[:, 1], spikes[:, 2], s = 80, color = 'k')
    plt.show()
    
def plotDistanceMatrix(distmat, title):
    ax = plt.axes()
    sns.heatmap(distmat, ax = ax)
    ax.set_title(title)
    plt.show()
    
def plotClusteringResult(width, height, title, label, labels_by_model, clustered_dataset, spikes):
    plt.figure(figsize=(width,height))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    plt.subplot(325)
    plt.title(title, fontsize='medium')
    for i in labels_by_model:
        plt.scatter(clustered_dataset[label == i , 0] , clustered_dataset[label == i , 1] , label = i)
    plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()
    
def calc_fed_euc_dist(sldm_array):
    combined_eucl = np.concatenate(sldm_array)
    # rows are s1,s2..sn while columns are datapoints
    print(combined_eucl)
    # computing the distance of distance (e.g.: meta-distance)
    # number of all samples * number of all samples
    return euclidean_distances(combined_eucl)

def regression_per_client(data, euc_dist_data_spike, regressor="Huber"):
    euc_dist_data = euclidean_distances(data).flatten()
    local_fed_dist = np.array(calc_fed_euc_dist([euc_dist_data_spike]).flatten()).reshape((-1,1))
    if regressor == "Huber":
        model = HuberRegressor().fit(local_fed_dist,euc_dist_data)
        return [model.coef_.item(),model.intercept_]
    if regressor == "Linear":
        model = LinearRegression().fit(local_fed_dist,euc_dist_data)
        return [model.coef_.item(),model.intercept_]
    if regressor == "TheilSen":
        model = TheilSenRegressor().fit(local_fed_dist,euc_dist_data)
        return [model.coef_.item(),model.intercept_]

# Generated dataset and use centroids as spike-in points
clustered_dataset, true_label, spikes = generate_clustered_dataset(dimension=12000,
                                                                   total_no_samples=500,
                                                                   no_of_cluster= 9, random_state=7)

# Plot the dataset and spike-in points for visualization
plot3dwithspike(width=9, height=6, datapoints = clustered_dataset, spikes=spikes, myLabel=true_label)

# Spliting the aggregated dataset for two participants
D1,D2 = np.array_split(clustered_dataset, 2)

######### Participant Based Computation ########
################################################
# # rows are s1,s2..sn while columns are datapoints
euc_dist_D1_spikes = euclidean_distances(D1,spikes)
# print("Spike local distance matrix of 1st participant: \n", euc_dist_D1_spikes)

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D2_spikes = euclidean_distances(D2,spikes)
# print("Spike local distance matrix of 2nd participant: \n", euc_dist_D2_spikes)

# Calculate and get slope and intercept for 1st participant's dataset  
slope_intercept_D1 = regression_per_client(data= D1,
                                           euc_dist_data_spike= euc_dist_D1_spikes,
                                           regressor="Huber")

# Calculate and get slope and intercept for 2nd participant's dataset
slope_intercept_D2 = regression_per_client(data= D2,
                                           euc_dist_data_spike= euc_dist_D2_spikes,
                                           regressor="Linear")

######### Cordinator Based Computation ########
################################################

def calc_fed_euc_dist(sldm_array):
    combined_eucl = np.concatenate(sldm_array)
    # computing the distance of distance i.e. meta-distance
    # number of all samples * number of all samples
    return euclidean_distances(combined_eucl)

def agglomerative_clustering(precomputed_dist_matrix, linkage='complete'):
    label = AgglomerativeClustering(affinity='precomputed', linkage=linkage).fit_predict(precomputed_dist_matrix)
    #Getting unique labels
    u_labels = np.unique(label)
    pred_label =  np.array(label).tolist()
    return label, u_labels, pred_label

def construct_global_Mx_Cx_matrix(MxCx,dataset_len_array):
    Mi,Ci = np.split(np.array(MxCx),2,axis=1)
    arrayMi=Mi.flatten()
    arrayCi=Ci.flatten()
    print("arrayMi: ",arrayMi)
    print("arrayCi: ",arrayCi)
    
    Mi_avg=np.average(arrayMi)
    Ci_avg=np.average(arrayCi)
    print("Average of slopes: ", Mi_avg)
    print("Average of constants: ", Ci_avg)
    print("array of number of vectors in the datasets, i.e., shape array: \n",dataset_len_array)
    
    #Placing the respective Mi of each datapoints and getting Mx matrix
    global_Mx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayMi, dataset_len_array)])
    #Placing the respective Ci of each datapoints and getting Cx matrix
    global_Cx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayCi, dataset_len_array)])
    print("Average of slope or coefficients i.e. Mi's: ", global_Mx)
    print("Average of constants or intercepts i.e. Ci's: ", global_Cx)
    # The zeroes in global slopes and constants matrix are replaced by Mi_avg and Ci_avg respectively 
    # They are used to calculate the predicted distance for cross-sectional data
    # For example: distance(a1,b5) where a1 and b5 belongs to different datasets
    global_Mx[global_Mx == 0] = Mi_avg
    global_Cx[global_Cx == 0] = Ci_avg
    print("Global coefficient or slope matrix: \n", global_Mx)
    print("Global constant or intercept matrix: \n", global_Cx)
    return global_Mx, global_Cx

def calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx):
    PGDM=np.add(np.multiply(global_Mx, global_fed_euc_dist),global_Cx)
    #As distance between same points is 0
    np.fill_diagonal(PGDM,0)
    # print("Predicted Global Distance Matrix: \n",PGDM)
    return PGDM

# Calculating global federated distance matrix
global_fed_euc_dist = calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

# Perform agglomerative_clustering using the above precomputed global federated distance matrix
label_gfdm, u_labels_gfdm, pred_label_gfdm = agglomerative_clustering(global_fed_euc_dist, linkage='complete')

#plotting the clustering results:
plotClusteringResult(width=15, height=15, 
                     title="Clustering with federated distance matrix",
                     label = label_gfdm,
                     labels_by_model=u_labels_gfdm,
                     clustered_dataset = clustered_dataset,
                     spikes = spikes)                   

# Reference: https://stackoverflow.com/questions/58069814/python-accuracy-check-giving-0-result-for-flipped-classification                
print("Adjusted Similarity score of the clustering with federated distance in (%) :", adjusted_rand_score(true_label, pred_label_gfdm)*100)
print("Adjusted mutual info score of the clustering with federated distance in (%) :", adjusted_mutual_info_score(true_label, pred_label_gfdm)*100)
print("Accuracy after clustering with federated distance in (%) :",accuracy_score(true_label, pred_label_gfdm)*100)

# Calculate global slopes and intercepts for each datapoints in clustered_dataset
MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)
global_Mx, global_Cx = construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])

# Calculating globally predicted distance matrix by applying
# the global slopes and intercept in the global federated distance matrix
global_pred_euc_dist = calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)


# Perform agglomerative_clustering using the above precomputed global predicted distance matrix
label = AgglomerativeClustering(affinity='precomputed', linkage='complete').fit_predict(global_pred_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_2 =  np.array(label).tolist()

# Plotting the clustering result
plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset[label == i , 0] , clustered_dataset[label == i , 1] , label = i)
plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

print("Adjusted Similarity score of the clustering with predicted distance in (%) :", adjusted_rand_score(true_label, pred_label_2)*100)
print("Adjusted mutual info score of the clustering with predicted distance in (%) :", adjusted_mutual_info_score(true_label, pred_label_2)*100)
print("Accuracy after clustering with predicted distance in (%) :",accuracy_score(true_label, pred_label_2)*100)

# Generating 3 heatmaps using actual, federated and predicted distance matrices
plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
plotDistanceMatrix(euclidean_distances(clustered_dataset), title="True Global Distance Matrix")
plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

print("Actual global distance matrix \n",euclidean_distances(clustered_dataset))
print("Predicted global distance matrix \n",global_pred_euc_dist)
print("The difference of the value in the matrices: \n", np.subtract(euclidean_distances(clustered_dataset), global_pred_euc_dist))

# Plotting attempt of the matrices
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111)

ax1.scatter(global_fed_euc_dist.flatten(), euclidean_distances(clustered_dataset).flatten(), s=10, c='b', marker="s", label='fed_with_true')
ax1.scatter(global_pred_euc_dist.flatten(), euclidean_distances(clustered_dataset).flatten(), s=10, c='r', marker="o", label='pred_with_true')
plt.legend(loc='upper left')
plt.show()

##### Observation: #####
########################
# Clustering doesnt depend on number of clients as there distances are all aggregated and fed to the clustering as distance matrix
# For 3 dimensional data and 2 clustering it gives very good result (100% accuracy or close to 1)
# For Linear Regression, HuberRegressor for 3 dimensional data and 3-9 clustering it gives poor result (5% to 60%)
# The accuracy of labeling decreases gradually with the increasing number of clustering defined in the dataset
# Difference between each value of actual and predicted distance matrices are in range of (0.00000001 to 3)
# Rate of difference might have some relation with the number of clusters or data samples
# Higher the dimension of the data points, less error on the predicted distance.

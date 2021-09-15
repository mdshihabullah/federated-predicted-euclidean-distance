import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.cluster import OPTICS
from sklearn.metrics import f1_score, calinski_harabasz_score, davies_bouldin_score
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


def plot3dwithspike(width, height, title, datapoints, spikes, myLabel=None) :
    plt.figure(figsize=(width,height))
    plt.title(title, fontsize='medium')
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

def generate_spikes_each_participant(dataset):
    dimension = dataset.shape[1]
    row_size = np.floor(np.sqrt(dimension)).astype(int) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else np.floor(np.sqrt(dataset.shape[0])).astype(int) 
    generated_spikes = np.random.uniform(low=np.min(dataset, axis=0),
                                         high=np.max(dataset, axis=0),
                                         size=(row_size, dimension))
    return generated_spikes

# Declare for experimental purpose
dimension=3
total_no_samples=120
no_of_cluster = 4

# Generated dataset and use centroids as spike-in points
clustered_dataset, true_label, centroids = generate_clustered_dataset(dimension=dimension,
                                                                   total_no_samples=total_no_samples,
                                                                   no_of_cluster= no_of_cluster, random_state=1)

# Plot the dataset and spike-in points for visualization
plot3dwithspike(width=9, height=6, title= "Clustering with actual labels", datapoints = clustered_dataset, spikes=centroids, myLabel=true_label)

# Spliting the aggregated dataset for two participants
D1,D2 = np.array_split(clustered_dataset, 2)

# Each participant generates random spike in points 
# which in production environment will be shared to coordinator for creating overall spike array
generated_spikes_D1 = generate_spikes_each_participant(D1)
generated_spikes_D2 = generate_spikes_each_participant(D2)

# Concatenate the locally generated spikes to create global spike array
generated_spikes = np.concatenate((generated_spikes_D1, generated_spikes_D2))

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D1_spikes = euclidean_distances(D1,generated_spikes)
# print("Spike local distance matrix of 1st participant: \n", euc_dist_D1_spikes)

# # rows are s1,s2..sn while columns are datapoints
euc_dist_D2_spikes = euclidean_distances(D2,generated_spikes)
# print("Spike local distance matrix of 2nd participant: \n", euc_dist_D2_spikes)

# Calculate and get slope and intercept for 1st participant's dataset  
slope_intercept_D1 = regression_per_client(data= D1,
                                           euc_dist_data_spike= euc_dist_D1_spikes,
                                           regressor="Huber")

# Calculate and get slope and intercept for 2nd participant's dataset  
slope_intercept_D2 = regression_per_client(data= D2,
                                           euc_dist_data_spike= euc_dist_D2_spikes,
                                           regressor="Linear")

def calc_fed_euc_dist(sldm_array):
    combined_eucl = np.concatenate(sldm_array)
    # rows are s1,s2..sn while columns are datapoints
    print(combined_eucl)
    # computing the distance of distance (e.g.: meta-distance)
    # number of all samples * number of all samples
    return euclidean_distances(combined_eucl)

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

global_true_euc_dist = euclidean_distances(clustered_dataset)

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
    plt.scatter(clustered_dataset[label == i , 0] , clustered_dataset[label == i , 1] , label = i)
plt.scatter(generated_spikes[:,0] , generated_spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

plot3dwithspike(width=9, height=6, title= "Clustering with true distance matrix", datapoints = clustered_dataset, spikes=generated_spikes, myLabel=pred_label_gtdm)

print("Adjusted Similarity score of the clustering with true distance in (%) :", adjusted_rand_score(true_label, pred_label_gtdm)*100)
print("Adjusted mutual info score of the clustering with true distance in (%) :", adjusted_mutual_info_score(true_label, pred_label_gtdm)*100)
print("F1 score after clustering with true distance:",f1_score(true_label, pred_label_gtdm, average='micro'))
print("Calinski-Harabasz Score: ", calinski_harabasz_score(global_true_euc_dist, pred_label_gtdm))
print("Davies-Bouldin Score: ", davies_bouldin_score(global_true_euc_dist, pred_label_gtdm))


global_fed_euc_dist = calc_fed_euc_dist([euc_dist_D1_spikes, euc_dist_D2_spikes])

label = OPTICS(metric='precomputed', n_jobs=-1).fit_predict(global_fed_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_gfdm =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset[label == i , 0] , clustered_dataset[label == i , 1] , label = i)
plt.scatter(generated_spikes[:,0] , generated_spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

plot3dwithspike(width=9, height=6, title= "Clustering with globally federated distance matrix", datapoints = clustered_dataset, spikes=generated_spikes, myLabel=pred_label_gfdm)

MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

global_Mx, global_Cx = construct_global_Mx_Cx_matrix(MxCx,[euc_dist_D1_spikes.shape[0], euc_dist_D2_spikes.shape[0]])
global_pred_euc_dist = calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)


label = OPTICS(metric='precomputed', n_jobs=-1).fit_predict(global_pred_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_2 =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset[label == i , 0] , clustered_dataset[label == i , 1] , label = i)
plt.scatter(generated_spikes[:,0] , generated_spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

plot3dwithspike(width=9, height=6, title= "Clustering with globally predicted distance matrix", datapoints = clustered_dataset, spikes=generated_spikes, myLabel=pred_label_2)

print("Adjusted Similarity score of the clustering with predicted distance in (%) :", adjusted_rand_score(pred_label_gtdm, pred_label_2)*100)
print("Adjusted mutual info score of the clustering with predicted distance in (%) :", adjusted_mutual_info_score(pred_label_gtdm, pred_label_2)*100)
print("F1 score after clustering with predicted distance:",f1_score(true_label, pred_label_2, average='micro'))
print("Calinski-Harabasz Score: ", calinski_harabasz_score(global_pred_euc_dist, pred_label_2))
print("Davies-Bouldin Score: ", davies_bouldin_score(global_pred_euc_dist, pred_label_2))

plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
plotDistanceMatrix(euclidean_distances(clustered_dataset), title="True Global Distance Matrix")
plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

print("Pearson correlation between true and predicted global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])
print("Pearson correlation between true and federated global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_fed_euc_dist.flatten())[0,1])
print("Pearson correlation between federated and predicted global matrices:", np.corrcoef(global_fed_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])
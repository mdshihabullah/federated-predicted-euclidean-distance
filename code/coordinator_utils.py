import numpy as np
from operator import itemgetter
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.linalg import block_diag
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import f1_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def generate_clustered_dataset(dimension,total_no_samples,no_of_cluster, random_state):
    """ Transform n-D array to specific number of dimension based array

    Args:
        dimension (int): Number of dimension the dataset to be reduced
        total_no_samples (int): Euclidean distance matrix between datapoints of a participant 
        no_of_cluster (int): Whether dimension is considered to how much spike points will be generated. Default value is False
        random_state (int): Reducing ratio for number of spike points i.e. the total number of spike points is divided by this value
        induce (int): Inducing ratio for number of spike points i.e. the total number of spike points is multiplied by this value
    Returns:
        clustered_dataset (n-D array): Scikit-learn PCA object
        true_label
        centroids (n-D array): n-D NumPy array of transformed dataset
    """
    clustered_dataset, true_label, centroids = make_blobs(n_samples=total_no_samples,
                                                          n_features=dimension, 
                                                          centers=no_of_cluster,
                                                          return_centers=True, 
                                                          random_state=random_state)
    return clustered_dataset, true_label, centroids


def optimal_no_of_cluster(euc_dist, range_n_clusters, threshold=0.95):
    silhouette_avg_array = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(euc_dist)
        silhouette_avg = silhouette_score(euc_dist, cluster_labels)
        silhouette_avg_array.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        if (silhouette_avg > threshold) or (n_clusters == np.ceil(len(range_n_clusters)/2 + 1).astype(int) and silhouette_avg < 0.2) :
            break
    return range_n_clusters[silhouette_avg_array.index(max(silhouette_avg_array))]


def perform_PCA(dimension, dataset):
    """ Transform n-D array to specific number of dimension based array

    Args:
        dimension (int): Number of dimension the dataset to be reduced
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        dimension_based (bool): Whether dimension is considered to how much spike points will be generated. Default value is False
        reduce (int): Reducing ratio for number of spike points i.e. the total number of spike points is divided by this value
        induce (int): Inducing ratio for number of spike points i.e. the total number of spike points is multiplied by this value
    Returns:
        pca (object): Scikit-learn PCA object
        (n-D array) : n-D NumPy array of transformed dataset
    """
    pca = PCA(n_components= dimension)
    return pca.fit_transform(dataset)

def perform_MDS(dimension, dataset):
    mds = MDS(n_components=dimension)
    return mds.fit_transform(dataset)

def get_participant_shape(shape_array):
    max_rows = max(shape_array,key=itemgetter(1))[0]
    min_rows = min(shape_array,key=itemgetter(1))[0]
    max_dim = max(shape_array,key=itemgetter(1))[1]
    min_dim = min(shape_array,key=itemgetter(1))[1]
    return max_rows, min_rows, max_dim, min_dim

def get_uniform_SLDM(SLDMi, min_dim):
    if SLDMi.shape[1] > min_dim:
        return perform_PCA(min_dim, SLDMi)
    else:
        return SLDMi

def calc_fed_euc_dist(sldm_array):
    """ Return the distance of the pairwise distances between each element of all the LSDM in list

    Args:
        sldm_array (list): Distance matrices each known as Local Spike Distance Matrix (LSDM) of all participants where each matrix 
                           is the pairwise euclidean distance between the spike points and the data points of respective participant
    
    Returns:
        (n-D array) : Global Federated Euclidean distance matrix between all datapoints of all participant(s)
    """
    combined_eucl = np.concatenate(sldm_array)
    # rows are datapoints while columns are S1,S2..Sn etc
    # computing the distance of distance (e.g.: meta-distance)
    # number of all samples * number of all samples
    return euclidean_distances(combined_eucl)

def construct_global_Mx_Cx_matrix(MxCx,dataset_len_array):
    """ Return two matrices containing the slopes and intercepts for each datapoints of all participants

    Args:
        MxCx (list): List of slopes and intercepts of all the participant(s)
        dataset_len_array (list): List of number of datapoints for each participant(s) 
    Returns:
        global_Mx (n-D array) : Matrix containing the slopes for each datapoints of all participants
        global_Cx (n-D array) : Matrix containing the intercepts for each datapoints of all participants
    """
    Mi,Ci = np.split(np.array(MxCx),2,axis=1)
    arrayMi=Mi.flatten()
    arrayCi=Ci.flatten()
    
    Mi_avg=np.average(arrayMi)
    Ci_avg=np.average(arrayCi)
    
    #Placing the respective Mi of each datapoints and getting Mx matrix
    global_Mx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayMi, dataset_len_array)])
    #Placing the respective Ci of each datapoints and getting Cx matrix
    global_Cx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayCi, dataset_len_array)])
    
    # The zeroes in global slopes and constants matrix are replaced by Mi_avg and Ci_avg respectively 
    # They are used to calculate the predicted distance for cross-sectional data
    # For example: distance(a1,b5) where a1 and b5 belongs to different datasets
    global_Mx[global_Mx == 0] = Mi_avg
    global_Cx[global_Cx == 0] = Ci_avg
    return global_Mx, global_Cx

def calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx):
    """ Return the distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM

    Args:
        global_Mx (n-D array) : Matrix containing the slopes for each datapoints of all participants
        global_fed_euc_dist (n-D array) : Matrix containing the distance of the pairwise distances between each element of all the participants' LSDM
        global_Cx (n-D array) : Matrix containing the intercepts for each datapoints of all participants
    Returns:
        PEDM (n-D array) : Distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM
    """
    # Multiply the slope with each value of FEDM followed by adding intercept with it 
    # to minimize error of the distance between two points
    PEDM=np.add(np.multiply(global_Mx, global_fed_euc_dist),global_Cx)
    #Fill the diagonal value to 0 as distance between same point is 0
    np.fill_diagonal(PEDM,0)
    # print("Predicted Global Distance Matrix: \n",PEDM)
    return PEDM

def plotDistanceMatrix(distmat, title):
    "Show heatmap of distance matrix"
    ax = plt.axes()
    sns.heatmap(distmat, ax = ax)
    ax.set_title(title)
    plt.savefig(title)

def unsupervised_evaluation_scores(dist_matrix, dist_matrix_name, expected_label, actual_label, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True):
    """ Evaluate and print the performance of clustering done by the distance matrix based on selected evaluation method, metric by providing expected and actual labels

    Args:
        dist_matrix (n-D array) : Distance matrix to evaluate the clustering done by it
        dist_matrix_name (str) : Name of the distance matrix
        expected_label (list) : List of integers containing the true label for each data points
        actual_label (list) : List of integers containing the actual label for each data points after performing clustering with the distance matrix
        adj_rand (bool) : Whether to calculate adjusted random score or not. Default value is True
        adj_mutual_info (bool) : Whether to calculate adjusted mutual info score or not. Default value is True
        f1 (bool) : Whether to calculate F1 score or not. Default value is True
        silhouette (bool) : Whether to calculate Silhouette score or not. Default value is False
        davies_bouldin (bool) : Whether to calculate Davies Bouldin score or not. Default value is True
    Returns:
        None
    """
    print(f"Adjusted similarity score of the clustering with {dist_matrix_name} in (%) :", adjusted_rand_score(expected_label, actual_label)*100) if adj_rand == True else 0
    print(f"Adjusted mutual info score of the clustering with {dist_matrix_name} in (%) :", adjusted_mutual_info_score(expected_label, actual_label)*100) if adj_mutual_info == True else 0
    print(f"F1 score after clustering with {dist_matrix_name}:",f1_score(expected_label, actual_label, average='micro')) if f1 == True else 0
    print(f"Silhouette score of {dist_matrix_name}:  ",silhouette_score(dist_matrix, actual_label, metric='precomputed')) if silhouette == True else 0
    print(f"Davies-Bouldin Score of {dist_matrix_name}: ", davies_bouldin_score(dist_matrix, actual_label)) if davies_bouldin == True else 0
    
def pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist):
    """ Calculate and print Pearson correlation coefficient value of PEDM w.r.t ADM, FEDM w.r.t ADM and FEDM w.r.t PEDM

    Args:
        global_true_euc_dist (n-D array) : True or actual pairwise distance between all the data points of all participant(s)
        global_fed_euc_dist (n-D array) : Matrix containing the distance of the pairwise distances between each element of all the participants' LSDM
        global_pred_euc_dist (n-D array) : Distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM
    Returns:
        None
    """
    print("Pearson correlation between true and predicted global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])
    print("Pearson correlation between true and federated global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_fed_euc_dist.flatten())[0,1])
    print("Pearson correlation between federated and predicted global matrices:", np.corrcoef(global_fed_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])

def spearman_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist):
    """ Calculate and print Spearman Rank correlation coefficient value of PEDM w.r.t ADM, FEDM w.r.t ADM and FEDM w.r.t PEDM

    Args:
        global_true_euc_dist (n-D array) : True or actual pairwise distance between all the data points of all participant(s)
        global_fed_euc_dist (n-D array) : Matrix containing the distance of the pairwise distances between each element of all the participants' LSDM
        global_pred_euc_dist (n-D array) : Distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM
    Returns:
        None
    """
    print("Spearman correlation between true and predicted global matrices:", spearmanr(global_true_euc_dist.flatten(),global_pred_euc_dist.flatten())[0])
    print("Spearman correlation between true and federated global matrices:", spearmanr(global_true_euc_dist.flatten(),global_fed_euc_dist.flatten())[0])
    print("Spearman correlation between federated and predicted global matrices:", spearmanr(global_fed_euc_dist.flatten(),global_pred_euc_dist.flatten())[0])

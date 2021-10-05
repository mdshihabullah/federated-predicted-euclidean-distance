import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
import matplotlib.pyplot as plt

def plot3dwithspike(width, height, title, datapoints, spikes, myLabel=None) :
    plt.figure(figsize=(width,height))
    plt.title(title, fontsize='medium')
    ax = plt.axes(projection='3d')
    ax.scatter3D(datapoints[:, 0], datapoints[:,1], datapoints[:,2], c=myLabel, marker='o',  s=15, edgecolor='k')
    ax.scatter3D(spikes[:, 0], spikes[:, 1], spikes[:, 2], s = 80, color = 'k')
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
    # computing the distance of distance (e.g.: meta-distance)
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

def generate_spikes_each_participant(dataset, dimension_based=False, reduce= 1):
    dimension = dataset.shape[1]
    if dimension_based == False:
        row_size = np.floor((np.sqrt(dimension))/reduce).astype(int) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else np.floor((np.sqrt(dataset.shape[0]))/reduce).astype(int) 
        generated_spikes = np.random.uniform(low=np.min(dataset, axis=0),
                                             high=np.max(dataset, axis=0),
                                             size=(row_size, dimension))
        return generated_spikes
    else:
        row_size = np.floor((np.sqrt(dimension))/reduce).astype(int)
        generated_spikes = np.random.uniform(low=np.min(dataset, axis=0),
                                             high=np.max(dataset, axis=0),
                                             size=(row_size, dimension))
        return generated_spikes


# (3rd answer) https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi
#For getting centroids as spikes when the label and clusters are known
def get_centroid_per_label(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])


#For getting centroids as spikes when the label and clusters are known
def get_spikes_from_centroid(df, col_name, unique_label):
    spikes= []
    for label in unique_label:
        spikes.append(get_centroid_per_label(df.loc[df[col_name] == label].drop(columns="Gene_ID").to_numpy(dtype='float64')))
    
    return np.array(spikes)
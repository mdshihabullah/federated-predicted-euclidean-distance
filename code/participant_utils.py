import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot3dwithspike(width, height, title, datapoints, spikes, myLabel=None) :
    plt.figure(figsize=(width,height))
    plt.title(title, fontsize='medium')
    ax = plt.axes(projection='3d')
    ax.scatter3D(datapoints[:, 0], datapoints[:,1], datapoints[:,2], c=myLabel, marker='o',  s=15, edgecolor='k')
    ax.scatter3D(spikes[:, 0], spikes[:, 1], spikes[:, 2], s = 80, color = 'k')
    plt.show()
    
    
def calc_fed_euc_dist(sldm_array):
    """ Return the distance of the pairwise distances between each element of all the LSDM in list

    Args:
        sldm_array (list): Distance matrices of all participants where each matrix 
                           is the pairwise euclidean distance between the spike points and the data points of respective participant
    
    Returns:
        (n-D array) : Global Federated Euclidean distance matrix between all datapoints of all participant(s)
    """
    combined_eucl = np.concatenate(sldm_array)
    # computing the distance of distance (e.g.: meta-distance)
    return euclidean_distances(combined_eucl)

def regression_per_client(data, euc_dist_data_spike, regressor="Huber"):
    """ Perform regression to correct FEDM. Get coefficient and intercept of regression model

    Args:
        data (n-D array): Euclidean distance matrix between datapoints of a participant 
        euc_dist_data_spike (n-D array): Distance matrix consisting pairwise euclidean distance 
                                         between spike points and data points
        regressor (str): Type of regressor to be used. Default value is "Huber".
                         Values can be "Huber", "Linear" or "TheilSen" respectively
    
    Returns:
        (list) : List consisting coefficient and intercept for each participant's dataset after performing regression.
    """
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
    return pca, pca.fit_transform(dataset)

def perform_PCA_inverse(pca, dataset_pca):
    """ Transform n-D array to back to original feature space

    Args:
        pca (object): Scikit-learn PCA object
        (n-D array) : n-D NumPy array of transformed dataset
    Returns:
        (n-D array) : n-D NumPy array of dataset inversely transformed back to original number of dimension
    """
    return pca.inverse_transform(dataset_pca)


def generate_spikes_using_PCA(dataset, dimension_based=False, reduce= 1, induce= 1):
    """ Generate uniformly distributed random spike points after performing PCA and inverse transforming it

    Args:
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        dimension_based (bool): Whether dimension is considered to how much spike points will be generated. Default value is False
        reduce (int): Reducing ratio for number of spike points i.e. the total number of spike points is divided by this value
        induce (int): Inducing ratio for number of spike points i.e. the total number of spike points is multiplied by this value
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points after applying 
                      inverse_transform() on the spike points with reduced dimension
    """
    dimension = dataset.shape[1]
    dimension_pca = dataset.shape[0] if dataset.shape[0] < dataset.shape[1] else dataset.shape[1]
    pca, dataset_pca = perform_PCA(dimension_pca, dataset)
    if dimension_based == False:
        no_of_spikes = np.floor((np.sqrt(dimension))/reduce).astype(int) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else np.floor((np.sqrt(dataset.shape[0]))/reduce).astype(int) 
        generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes*induce, dimension_pca))
        
        return perform_PCA_inverse(pca, generated_spikes)
    else:
        no_of_spikes = np.floor((np.sqrt(dimension))/reduce).astype(int)
        generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes*induce, dimension_pca))
        return perform_PCA_inverse(pca, generated_spikes)

def generate_spikes_using_PCA_and_variance(dataset, variance=0.90, dimension_based=False, reduce= 1, induce= 1):
    """ Generate uniformly distributed random spike points after performing PCA and inverse transforming it

    Args:
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        variance (float): Amount of variance to be covered after performing PCA in the dataset. 
                          Value range from 0.0 to 1.0
        dimension_based (bool): Whether dimension is considered to how much spike points will be generated
        reduce (int): Reducing ratio for number of spike points i.e. the total number of spike points is divided by this value
        induce (int): Inducing ratio for number of spike points i.e. the total number of spike points is multiplied by this value
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points after applying 
                      inverse_transform() on the spike points with reduced dimension
    """
    dimension = dataset.shape[1]
    pca, dataset_pca = perform_PCA(variance, dataset)
    if dimension_based == False:
        no_of_spikes = np.floor((np.sqrt(dimension))/reduce).astype(int) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else np.floor((np.sqrt(dataset.shape[0]))/reduce).astype(int) 
        generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes*induce, dataset_pca.shape[1]))
        
        return perform_PCA_inverse(pca, generated_spikes)
    else:
        no_of_spikes = np.floor((np.sqrt(dimension))/reduce).astype(int)
        generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes*induce, dataset_pca.shape[1]))
        return perform_PCA_inverse(pca, generated_spikes)

def generate_n_spikes_using_PCA(dataset, no_of_spikes ):
    """ Generate N number of uniformly distributed random spike points after performing PCA and inverse transforming it

    Args:
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        no_of_spikes (int): Number of spike points to be generated and returned
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points after applying 
                      inverse_transform() on the spike points of reduced dimension
    """
    dimension_pca = dataset.shape[0] if dataset.shape[0] < dataset.shape[1] else dataset.shape[1]
    pca, dataset_pca = perform_PCA(dimension_pca, dataset)
    generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes, dimension_pca))
        
    return perform_PCA_inverse(pca, generated_spikes)

def generate_n_spikes_using_variance(variance, dataset, no_of_spikes ):
    """ Generate N number of uniformly distributed random spike points after performing PCA and inverse transforming it

    Args:
        variance (float): Amount of variance to be covered after performing PCA in the dataset. 
                          Value range from 0.0 to 1.0
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        no_of_spikes (int): Number of spike points to be generated and returned
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points after applying 
                      inverse_transform() on the spike points of reduced dimension
    """
    pca, dataset_pca = perform_PCA(variance, dataset)
    generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                             high=np.max(dataset_pca, axis=0),
                                             size=(no_of_spikes, dataset_pca.shape[1]))
        
    return perform_PCA_inverse(pca, generated_spikes)


def generate_spikes_with_centroids(dataset, centroids, reduce= 1, induce= 1):
    """ Generate uniformly distributed random spike points along with centroids

    Args:
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        centroids (n-D array): centroids returned which are considered as spike points for all the participants
        reduce (int): Reducing ratio for number of spike points i.e. the total number of spike points is divided by this value
        induce (int): Inducing ratio for number of spike points i.e. the total number of spike points is multiplied by this value
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points along with centroids
                      after applying inverse_transform() on the spike points with reduced dimension
    """
    dimension = dataset.shape[1]
    dimension_pca = dataset.shape[0] if dataset.shape[0] < dataset.shape[1] else dataset.shape[1]
    pca, dataset_pca = perform_PCA(dimension_pca, dataset)
    no_of_spikes = abs(np.floor(((np.sqrt(dimension))*induce)/reduce).astype(int) + centroids.shape[0]) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else abs(np.floor((np.sqrt(dataset.shape[0])*induce)/reduce).astype(int) + centroids.shape[0])
    generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                         high=np.max(dataset_pca, axis=0),
                                         size=(no_of_spikes*induce, dimension_pca))
        
    return np.concatenate((perform_PCA_inverse(pca, generated_spikes), centroids))


def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]
    
def generate_N_spikes_with_centroids(dataset, centroids, size):
    """ Generate uniformly distributed random spike points along with centroids

    Args:
        dataset (n-D array): Euclidean distance matrix between datapoints of a participant 
        centroids (n-D array): centroids returned which are considered as spike points for all the participants
        size (int): Amount of spike points to be returned
    Returns:
        (n-D array) : n-D NumPy array of generated random spike points along with centroids
                      after applying inverse_transform() on the spike points with reduced dimension
    """
    dimension = dataset.shape[1]
    pca, dataset_pca = perform_PCA(0.90, dataset)
    no_of_spikes = abs(np.floor(((np.sqrt(dimension))*1)/1).astype(int) + centroids.shape[0]) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else abs(np.floor((np.sqrt(dataset.shape[0])*1)/1).astype(int) + centroids.shape[0])
    generated_spikes = np.random.uniform(low=np.min(dataset_pca, axis=0),
                                         high=np.max(dataset_pca, axis=0),
                                         size=(no_of_spikes, dataset_pca.shape[1]))
        
    spikes = np.concatenate((perform_PCA_inverse(pca, generated_spikes), centroids))

    return random_sample(spikes, size)

################ EXTRA METHODS THAT WERE NEVER USED ####################

# (3rd answer) https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi
#For getting centroids as spikes when the label and clusters are known
def get_centroid_per_label(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])


#For getting centroids as spikes when the label and clusters are known
def get_spikes_from_centroid(df, col_name, unique_label):
    spikes= []
    for label in unique_label:
        spikes.append(get_centroid_per_label(df.loc[df[col_name] == label].drop(columns=col_name).to_numpy(dtype='float64')))
    
    return np.array(spikes)

def generate_spikes_each_participant(dataset, dimension_based=False, permute=True, reduce= 1, induce= 1):
    dimension = dataset.shape[1]
    dimension_pca = dataset.shape[0] if dataset.shape[0] < dataset.shape[1] else dataset.shape[1]
    pca, dataset_pca = perform_PCA(dimension_pca, dataset)
    low_pca = np.min(dataset_pca, axis=0)
    high_pca = np.max(dataset_pca, axis=0)
    
    low_inverse = perform_PCA_inverse(pca, low_pca)
    high_inverse = perform_PCA_inverse(pca, high_pca)
    if dimension_based == False:
        no_of_spikes = np.floor(((np.sqrt(dimension))*induce)/reduce).astype(int) if np.floor(np.sqrt(dimension)).astype(int) < np.floor(np.sqrt(dataset.shape[0])).astype(int) else np.floor((np.sqrt(dataset.shape[0])*induce)/reduce).astype(int)
        divlen = np.subtract(high_inverse,low_inverse)/no_of_spikes
        evenspikes = []
        for i in range(0,no_of_spikes):
            divspikearray = np.random.uniform(low=np.add(low_inverse,i*divlen),
                                             high=np.add(low_inverse,(i+1)*divlen),
                                             size=dimension)
            evenspikes.append(divspikearray)

        generated_spikes = np.array(evenspikes)
        return np.random.permutation(generated_spikes) if permute else generated_spikes
    else:
        no_of_spikes = np.floor((np.sqrt(dimension))/reduce).astype(int)
        divlen = (np.subtract(high_inverse, low_inverse)) / no_of_spikes
        evenspikes = []
        for i in range(0, no_of_spikes):
            divspikearray = np.random.uniform(low=low_inverse + (i * divlen),
                                              high = low_inverse + ((i + 1) * divlen),
                                              size = (1, dimension))
            evenspikes.append(divspikearray)

        generated_spikes = np.array(evenspikes)
        return generated_spikes


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
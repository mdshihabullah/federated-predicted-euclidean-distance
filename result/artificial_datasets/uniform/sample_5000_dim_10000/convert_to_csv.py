import os
import csv
from pathlib import Path
import re
from fnmatch import fnmatch

def get_method_name(fileName):
    if fnmatch(fileName, '*FEDM_PEDM_using_N_centroids_and_generated_spikes_blob_*') == True:
        return "random with centroids"
    elif fnmatch(fileName, '*FEDM_PEDM_using_N_centroids_blob_*') == True:
        return "centroids"
    else:
        return "random"    


os.chdir(r'C:\Users\christina\Documents\GitHub\federated-distance-calculation\result\artificial_datasets\uniform\sample_5000_dim_10000')
with open('sample_5000_dim_10000_results.csv', 'w', newline='') as out_file:
     csv_out = csv.writer(out_file)
     csv_out.writerow(["dataset","no_of_spikes","pearson_ADM_PEDM","pearson_ADM_FEDM","pearson_PEDM_FEDM","spearman_ADM_PEDM","spearman_ADM_FEDM","spearman_PEDM_FEDM","method"])
     for fileName in Path('.').glob('*.txt'):
         lines = fileName.read_text().splitlines()
         print(fileName)      
         no_of_spikes = re.findall(r'^\D*(\d+)', lines[0])[0]
         pearson_ADM_PEDM = re.findall("[+-]?\d+\.\d+", lines[1])[0]
         pearson_ADM_FEDM = re.findall("[+-]?\d+\.\d+", lines[2])[0]
         pearson_PEDM_FEDM = re.findall("[+-]?\d+\.\d+", lines[3])[0]
         spearman_ADM_PEDM = re.findall("[+-]?\d+\.\d+", lines[4])[0]
         spearman_ADM_FEDM = re.findall("[+-]?\d+\.\d+", lines[5])[0]
         spearman_PEDM_FEDM = re.findall("[+-]?\d+\.\d+", lines[6])[0]
         method_name = get_method_name(str(fileName))
         csv_out.writerow(["sample_5000_dim_10000", no_of_spikes, pearson_ADM_PEDM, pearson_ADM_FEDM, pearson_PEDM_FEDM, spearman_ADM_PEDM, spearman_ADM_FEDM, spearman_PEDM_FEDM, method_name])
        #  csv_out.writerow([no_of_spikes])
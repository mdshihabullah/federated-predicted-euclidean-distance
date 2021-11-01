import os
import csv
from pathlib import Path
import re
os.chdir(r'C:\Users\christina\Documents\GitHub\federated-distance-calculation\result\GSE84433\using_N_generated_spikes_and_centroids')
with open('GSE84433_N_generated_spikes_and_centroids.csv', 'w', newline='') as out_file:
     csv_out = csv.writer(out_file)
     csv_out.writerow(["no_of_spikes","pearson_ADM_PEDM","pearson_ADM_FEDM","pearson_PEDM_FEDM","spearman_ADM_PEDM","spearman_ADM_FEDM","spearman_PEDM_FEDM"])
     for fileName in Path('.').glob('*.txt'):
         lines = fileName.read_text().splitlines()      
         no_of_spikes = re.findall(r'^\D*(\d+)', lines[0])[0]
         pearson_ADM_PEDM = re.findall("[+-]?\d+\.\d+", lines[1])[0]
         pearson_ADM_FEDM = re.findall("[+-]?\d+\.\d+", lines[2])[0]
         pearson_PEDM_FEDM = re.findall("[+-]?\d+\.\d+", lines[3])[0]
         spearman_ADM_PEDM = re.findall("[+-]?\d+\.\d+", lines[4])[0]
         spearman_ADM_FEDM = re.findall("[+-]?\d+\.\d+", lines[5])[0]
         spearman_PEDM_FEDM = re.findall("[+-]?\d+\.\d+", lines[6])[0]
         csv_out.writerow([no_of_spikes, pearson_ADM_PEDM, pearson_ADM_FEDM, pearson_PEDM_FEDM, spearman_ADM_PEDM, spearman_ADM_FEDM, spearman_PEDM_FEDM])
        #  csv_out.writerow([no_of_spikes])
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"]=(25,25)
# sns.set_theme(style="whitegrid")

boxplot_file_path = Path(__file__).parent / "../result/GSE84426_results.csv"
boxplot_results = pd.read_csv(boxplot_file_path)

artificial_non_uniform_file_path = Path(__file__).parent / "../result/non_uniform_artificial_data_results.csv"
artificial_non_uniform_results = pd.read_csv(artificial_non_uniform_file_path)

artificial_uniform_file_path = Path(__file__).parent / "../result/uniform_artificial_data_results.csv"
artificial_uniform_results = pd.read_csv(artificial_uniform_file_path)

boxplot_GSE84433_N_centroids_file_path = Path(__file__).parent / "../result/GSE84433_N_centroids.csv"
boxplot_GSE84433_N_centroids_results = pd.read_csv(boxplot_GSE84433_N_centroids_file_path)

boxplot_GSE84433_N_generated_spikes_file_path = Path(__file__).parent / "../result/GSE84433_N_generated_spikes.csv"
boxplot_GSE84433_N_generated_spikes_results = pd.read_csv(boxplot_GSE84433_N_generated_spikes_file_path)

boxplot_GSE84433_N_generated_spikes_and_centroids_file_path = Path(__file__).parent / "../result/GSE84433_N_generated_spikes_and_centroids.csv"
boxplot_GSE84433_N_generated_spikes_and_centroids_results = pd.read_csv(boxplot_GSE84433_N_generated_spikes_and_centroids_file_path)


lineplot_file_path = Path(__file__).parent / "../result/GSE84426_lineplot_results.csv"
lineplot_results = pd.read_csv(lineplot_file_path)

g = sns.FacetGrid(data=artificial_non_uniform_results, col="dataset")
g.map_dataframe(sns.pointplot, 
      "no_of_spikes", "spearman_ADM_PEDM",
      hue="method", palette='deep')
g.add_legend()

#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient"
#              r'$\mathit{r}$ (ADM, PEDM)'
#              r'$\rho$ (ADM, PEDM)'
g.set_xlabels("Number of spike points", labelpad=20)
g.set_ylabels(r'$\rho$ (ADM, PEDM)', labelpad=20)
plt.show()

# sns.catplot(x='no_of_spikes', y='pearson_ADM_PEDM', hue='method', data=artificial_non_uniform_results, kind='point', col='dataset')
# plt.show()

def showCatPlot(x_values, y_values, hue_values, data, kind, col, title, x_label, y_label):
    ax = sns.catplot(x=x_values, y=y_values, hue=hue_values, data=data, kind=kind, col=col)
    # Customize the axes and title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel(y_label, labelpad=20)
    plt.show()

def showBoxPlot(x_values,y_values,hue_values,data,title,x_label,y_label):
    ax = sns.boxplot(x=x_values, y=y_values, hue=hue_values, data=data)
    # Customize the axes and title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel(y_label, labelpad=20)
    plt.show()

def showPointPlot(x_values,y_values,hue_values,data,title,x_label,y_label):
    ax = sns.pointplot(x=x_values, y=y_values, hue=hue_values, data=data)
    # Customize the axes and title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel(y_label, labelpad=20)
    plt.show()

def showLinePlot(x_values,y_values,hue_values,data,title,x_label,y_label):
    ax = sns.lineplot(data=data, x=x_values, y=y_values, hue=hue_values)
    # Customize the axes and title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel(y_label, labelpad=20)
    plt.show()


# showLinePlot(x_values="no_of_spikes",
#             y_values="correlation_values",
#             hue_values="correlation_entities",
#             data=lineplot_results,
#             title= "Overall Pearson and Spearman Correlation Coefficient",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showPointPlot(x_values="no_of_spikes",
#             y_values="pearson_ADM_PEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Pearson correlation between Aggregated and Predicted Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showBoxPlot(x_values="no_of_spikes",
#             y_values="pearson_ADM_PEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Pearson correlation between Aggregated and Predicted Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showPointPlot(x_values="no_of_spikes",
#             y_values="pearson_ADM_FEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Pearson correlation between Aggregated and Federated Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showBoxPlot(x_values="no_of_spikes",
#             y_values="pearson_ADM_FEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Pearson correlation between Aggregated and Federated Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showPointPlot(x_values="no_of_spikes",
#             y_values="spearman_ADM_PEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Spearman correlation between Aggregated and Predicted Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showBoxPlot(x_values="no_of_spikes",
#             y_values="spearman_ADM_PEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Spearman correlation between Aggregated and Predicted Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showPointPlot(x_values="no_of_spikes",
#             y_values="spearman_ADM_FEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Spearman correlation between Aggregated and Federated Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showBoxPlot(x_values="no_of_spikes",
#             y_values="spearman_ADM_FEDM",
#             hue_values="method",
#             data=boxplot_results,
#             title= "Spearman correlation between Aggregated and Federated Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")

# showCatPlot(x_values="no_of_spikes",
#             y_values="pearson_ADM_PEDM",
#             hue_values="method",
#             data=artificial_non_uniform_results,
#             kind='box',
#             col='dataset',
#             title= "Pearson correlation between Aggregated and Predicted Euclidean Distance Matrices",
#             x_label="Number of spike points", 
#             y_label="Correlation Coefficient")
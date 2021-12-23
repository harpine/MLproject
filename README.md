# MLproject - Miniproject BIO-322, J.Brea
## Helena Binková & Aline Brunner, team HBBA

The aim of the project is to predict the probability of rain on the next day in Pully, given the meteorogical measurements in different measure stations of Switzerland.
This project takes the form of a kaggle competition. The data are available here (with access only): https://www.kaggle.com/c/bio322-will-it-rain-tomorrow/overview.
The submissions should be a CSV file containing the probability of raining for each row of the test data. 

## Requirements 
This programm has been coded in Julia, using version 1.7.0 and uses the following packages in the given version:

- CSV v0.9.11
- CategoricalArrays v0.10.2
- CategoricalDistributions v0.1.3
- DataFrames v1.3.0
- Flux v0.12.8
- GLMNet v0.7.0
- LIBSVM v0.6.0
- MLJ v0.16.11
- MLJDecisionTreeInterface v0.1.3
- MLJFlux v0.2.5 `https://github.com/FluxML/MLJFlux.jl.git#2a8d193`
- MLJLIBSVMInterface v0.1.4
- MLJLinearModels v0.5.7
- MLJMultivariateStatsInterface v0.2.2
- NearestNeighborModels v0.1.6
- PlotlyJS v0.18.8
- Plots v1.25.2
- StatsPlots v0.14.29
- Random


## Github repository organization 
### dependencies files
The Manifest.toml and Project.toml give information about the versions of dependencies required for the project. 
You can install the dependencies with the command pkg.instantiate(). This command is included in the initialization script (see **Execution**, below). 

### Scripts
- `comparison_machines.jl`: permits to create the boxplots to visually compare the machines. 
- `data_preprocessing.jl`: preprocesses the data and puts them in the datasets folder, in serialized format. First, it fills the missing training data. Then it creates a standardized training set, and a normalized one. It also computes a regularized dataset with less predictors, using the Lasso. The same transformations are applied to the test set to obtain corresponding sets. This file uses functions from `sorting_data_regularization.jl`. All the pre-processed data are serialized and saved in the `datasets` folder. 
- `datasets.jl`: retrieves the data from serialization, so that they don't have to be preprocessed at each run. 
- `initialization.jl`: setup the environment from the Manifest.toml and creates the initial folders through `utilities.jl`.
- `losses.jl`: plots the learning curve of the neural network on a validation set.
- `machines_run.jl`: provides the functions and configurations to run the machines from the main file: `script.jl`.
- `machines.jl`: contains the five machines that are cited in the report (multiple logistic regression, knn, random forest, neural network with one hidden layer and neural network with two hidden layers). Each machine is wrapped in a function, which tunes the model, saves its resulting machine, shows the fitted parameters, the mean AUC measurements during cross validation, the final error rate on training set and the final AUC. Then it writes the predictions in the `ouptuts` folders, the statistics in the corresponding folder and saves the plot of the machine report.
- `plot_lasso.jl`: permits to show a Lasso path, with the 18 firsts predictors in order of importance. 
- `save_statistics.jl`: contains different functions to save statistics of the tuned machine, depending on the machine type. The functions with the suffix *old* permit to save machines statistics that are already saved and for which we don't have the model stored anymore. 
- `script.jl`: main file of the program. Permits to run the machines with command line given different arguments, using ArgParse. It permits to run all machines, only the best, or only the given one. 
- `sorting_data_regularization.jl`: provides function to compute the regularized and the normalized datasets. The functions are used in data_preprocessing.jl
- `utilities.jl`: configuration file. All folder and file names and paths are configurable by replacing the given string. It also provides three utility functions to write CSV files in the right folder. 
- `visualization.jl`: shows the training data in two different forms: a pca biplot and a correlation plot.


### Folders
#### Available folders on github:
- `best_machines`: contains all best machines results per type, which are cited in the report
- `figures`: contains report figures
- `src`: contains whole code 

#### Created folders:
These are the default folders, if the name and location haven't been changed in `utilities.jl`.

- `datasets`: folder containing the initial datasets and where the preprocessing datasets are going to be put. If a path is specified for the original data, this folder contains only the preprocessed ones.
- `losses`: folder containing the graph of the learning curves for the neuralnetwork.
- `machines`: folder containing the machines themselves, to be able to retrieve them to do further measurement or to use them without training the again.
- `outputs`: folder containing the output, ready to be submitted on Kaggle. 
- `plots`: folder containing the machine report plots. 
- `statistics`: folder containing the statistics related to the trained machine

## Execution


Be aware that the following commands have to be run from a bash terminal, as we use command-line argument parsing.

From the `MLproject` folder, run
```
~/path/to/<Julia directory>/bin/julia src/initialization.jl
```
Have a look at the paths and file names defined in `utilities.jl`. Feel free to change them to your conveniance, in particular the `original_dataset_folder_name` which corresponds to the relative path to the datasets (training & test) folder. In case you don't already have a folder containing them, an empty folder (`datasets`) has been created in the actual repository to allow you to put your data inside. 

In order to run the tuning of the machines, please run the following command 

```
~/path/to/<Julia directory>/bin/julia src/script.jl --machines MACHINES --machine_subname MACHINE_SUBNAME
```
Where:
- `MACHINES` is the type of machine that will be run. The different possibilities are: `all` (run all in report cited machines), `best` (run the two best submission on kaggle), `logistic_reg`, `knn`, `random_forest`, `short_neuralnetwork` or `mlp_neuralnetwork`. Be aware that certain networks are very long to run! *Default is "best"*
- `Machine_subname` is the subname of the machine. The given string will be concatenated to the name of the type of the machine. This is to specify only if you want to run a specific machine. 

To plot the boxplots of the first best retained models, run:
```
~/path/to/<Julia directory>/bin/julia src/comparison_machines.jl
```
To obtain the lasso plot, run:
```
~/path/to/<Julia directory>/bin/julia src/plot_lasso.jl
```
To obtain the pca and correlation plot, run:
```
~/path/to/<Julia directory>/bin/julia src/visualization.jl
```




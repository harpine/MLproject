# MLproject

The aim of the project is to predict the probability of rain on the next day in Pully, given the meteorogical measurements in different measure stations of Switzerland.
This project takes the form of a kaggle competition. The data are available here (with access only): \url{https://www.kaggle.com/c/bio322-will-it-rain-tomorrow/overview}.
The submissions should be a CSV file containing the probability of raining for each row of the test data. 

# Requirements 
This programm has been coded in Julia, using version 1.7.0

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


# Scripts organization 

## Folders
-- where to put data

# Execution
- (reproducibility of results)
- user interface 
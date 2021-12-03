using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using MLCourse
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels

data = CSV.read(joinpath(@__DIR__, "datasets", "trainingdata.csv"), DataFrame)
training_data = data[1:1000,:]
validation_data = data
coerce!(training_data, :precipitation_nextday => Multiclass)
training_dropped = dropmissing(training_data)
training_dropped_x = select(training_dropped, Not(:precipitation_nextday))
training_dropped_y = training_dropped.precipitation_nextday
training_filled = MLJ.transform(fit!(machine(FillImputer(), training_data)), training_data)
training_filled_x = select(training_filled, Not(:precipitation_nextday))
training_filled_y = training_dropped.precipitation_nextday


#Test set
test_data = CSV.read(joinpath(@__DIR__, "datasets", "testdata.csv"), DataFrame)
coerce!(test_data, :precipitation_nextday => Multiclass)
test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)


Random.seed!(3)
"""
@df training_filled corrplot([:ABO_sunshine_1 :ABO_delta_pressure_1 :ABO_radiation_1 :ABO_wind1 :ABO_wind_direction_1],
                     grid = false, fillcolor = cgrad(), size = (700, 700))
"""


## VISUALIZATION:
# datframe on pluto - not on git
# plot rain in function of different predictors
# correlation-plot (weatherdata série 2)
# are there missing data (true/false) -> how many? drop, or mean? - série 6b

# standardization needed?  (série 6b)
# vizualisation of conditional output ? (maybe, to look)

## MACHINES: 
#naming: ex: mach_logistic_class_m or mach_logistic_class_f

# -> create machine, 
#graph prediction/ground truths, 
#calculation of errors (confusion matrix, roc, auc, losses (which?) [série 3 - function losses, série 4 - mnisterrorrate]), 
# cross-validation (included in self-tuning model), série 5

#bias-variance decomposition: to check overfitting - série 4 see graph

# Data:
# transform in categorical data (coerce - série 3)
# TYPES:

## LINEAR

# Multiple Logistic Regression (using MLJ, MLJLinearModels : LogisticClassifier) - série 3
# Flexible methods : polynomial classification, useful? ( LogisticClassifier select ) - série 4 (+graph flexibility classification)
# K-nearest neighbor Classification (KNNClassifier) - série 4 (graph flexibility)
# RidgeRegressor: tune penalty of LogicticClassifier - série 6a (lasso not used)
# vector features - série 6b
# gradient descent (stochastic) for logistic regression (opt = ADAMW() or ADAM() ? ) - série 7
# early stopping - série 7

##NONLINEAR

# Leanred features vector - série 8
# NeuralNetworkClassifier (MLJFlux)- série 8 (CLassification with MLPs)
# Support Vector Machines 
# 

# what is MultinomialClassifier , is it useful?

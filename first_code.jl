
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using MLCourse
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels

training_data = CSV.read(joinpath(@__DIR__, "datasets", "trainingdata.csv"), DataFrame)
training_data = coerce()String.(training_data.label, Binary)
training_wo_missing = dropmissing(training_data)
training_filled = MLJ.transform(fit!(machine(FillImputer(), training_data)), training_data)
## VISUALIZATION:
# datframe on pluto - not on git
# plot rain in function of different predictors
# correlation-plot (weatherdata série 2)
# are there missing data (true/false) -> how many? drop, or mean? - série 6b

# standardization needed?  (série 6b)
# vizualisation of conditional output ? (maybe, to look)


## MACHINES: 

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

logistic_regression_missing = machine(LogisticClassifier(penalty = :none), training, trainin) |> fit!



##NONLINEAR

# Leanred features vector - série 8
# NeuralNetworkClassifier (MLJFlux)- série 8 (CLassification with MLPs)
# Support Vector Machines 
#

# what is MultinomialClassifier , is it useful?

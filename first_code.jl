using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface

training_data = CSV.read(joinpath(@__DIR__, "datasets", "trainingdata.csv"), DataFrame)

coerce!(training_data, :precipitation_nextday => Multiclass)
training_dropped = dropmissing(training_data)
training_dropped_x = select(training_dropped, Not(:precipitation_nextday))
training_dropped_y = training_dropped.precipitation_nextday
standardizer_mach_dropped = fit!(machine(Standardizer(features = Symbol[:ALT_sunshine_4, :ZER_sunshine_1], ignore = true),  training_dropped_x)) #, verbosity = 2) #features ignore to retrieve too small variances
training_dropped_x_std = MLJ.transform(standardizer_mach_dropped, training_dropped_x)

training_dropped_x_mlp = coerce!(training_dropped_x, Count => MLJ.Continuous)

training_filled = MLJ.transform(fit!(machine(FillImputer(), training_data)), training_data)
training_filled_x = select(training_filled, Not(:precipitation_nextday))
training_filled_y = training_filled.precipitation_nextday
standardizer_mach_filled = fit!(machine(Standardizer(features = Symbol[:ALT_sunshine_4], ignore = true), training_filled_x)) #, verbosity = 2) #features ignore to retrieve too small variances
training_filled_x_std = MLJ.transform(standardizer_mach_filled, training_filled_x)

#training_filled_x_std = MLJ.transform(fit!(machine(Standardizer(), training_filled_x)), training_filled_x) -> à réfléchir si on veut train un nouveau sur les filled ou utiliser l'autre. et si oui, lequel on utilise pour standardiser le test.

#write_csv("test_std.csv", training_filled_x_std)

#Test set
test_data = CSV.read(joinpath(@__DIR__, "datasets", "testdata.csv"), DataFrame)
coerce!(test_data, :precipitation_nextday => Multiclass)
test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data) # We have to fill the missing datas, because we want a prediction for all existing datas.
test_data_std = MLJ.transform(standardizer_mach_filled, test_data)

output_folder_name = "outputs"
output_folder = joinpath(@__DIR__, output_folder_name)
mkpath(output_folder)

machines_folder_name = "machines"
machines_folder = joinpath(@__DIR__, machines_folder_name)
mkpath(machines_folder)

plots_folder_name = "plots"
plots_folder = joinpath(@__DIR__, plots_folder_name)
mkpath(plots_folder)

stat_folder_name = "statistics"
stat_folder = joinpath(@__DIR__, stat_folder_name)
mkpath(stat_folder)

function write_csv(output_file_name, dataframe)
    CSV.write(joinpath(output_folder, output_file_name), dataframe)
end

function write_stat(output_file_name, dataframe)
    CSV.write(joinpath(stat_folder, output_file_name), dataframe)
end

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

# Learned features vector - série 8
# NeuralNetworkClassifier (MLJFlux)- série 8 (CLassification with MLPs) -> Aline 
# Support Vector Machines - série 9 -> Helena
# 

# what is MultinomialClassifier , is it useful?

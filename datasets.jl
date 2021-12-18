using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface
include("./utilities.jl")

training_filled_x = deserialize(joinpath(dataset_folder, training_x_name))
training_filled_x_std = deserialize(joinpath(dataset_folder, training_x_std_name))
training_filled_y = deserialize(joinpath(dataset_folder, training_y_name))

regularized_training_filled_x = deserialize(joinpath(dataset_folder, regularized_training_filled_x_name))
regularized_training_filled_x_std = deserialize(joinpath(dataset_folder, regularized_training_filled_x_std_name))
regularized_training_filled_x_norm = deserialize(joinpath(dataset_folder, regularized_training_filled_x_norm_name))

test_data = deserialize(joinpath(dataset_folder, test_name))
test_data_std = deserialize(joinpath(dataset_folder, test_std_name))
regularized_test_data_norm = deserialize(joinpath(dataset_folder, regularized_test_norm_name))
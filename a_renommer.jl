using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface


training_filled_x = CSV.read(joinpath(dataset_folder, training_x_name), DataFrame)
training_filled_x_std = CSV.read(joinpath(dataset_folder, training_x_std_name), DataFrame)
training_filled_y = CSV.read(joinpath(dataset_folder, training_y_name), DataFrame)

regularized_training_filled_x = CSV.read(joinpath(dataset_folder, regularized_training_filled_x_name), DataFrame)
regularized_training_filled_x_std = CSV.read(joinpath(dataset_folder, regularized_training_filled_x_std_name), DataFrame)
regularized_training_y = CSV.read(joinpath(dataset_folder, regularized_training_y_name), DataFrame)

test_data = CSV.read(joinpath(dataset_folder, test_name), DataFrame)
test_data_std = CSV.read(joinpath(dataset_folder, test_std_name), DataFrame)
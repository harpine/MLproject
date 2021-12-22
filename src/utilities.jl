using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface, Serialization, ArgParse

include("./save_statistics.jl")

# Name of dataset files
training_data_name = "trainingdata.csv"
test_data_name = "testdata.csv"

# Folders name and location 
dataset_folder_name = joinpath( "..","datasets")
dataset_folder = joinpath(@__DIR__, dataset_folder_name)
mkpath(dataset_folder)

output_folder_name = joinpath("..", "outputs")
output_folder = joinpath(@__DIR__, output_folder_name)
mkpath(output_folder)

machines_folder_name = joinpath("..", "machines")
machines_folder = joinpath(@__DIR__, machines_folder_name)
mkpath(machines_folder)

plots_folder_name = joinpath("..", "plots")
plots_folder = joinpath(@__DIR__, plots_folder_name)
mkpath(plots_folder)

stat_folder_name = joinpath("..", "statistics")
stat_folder = joinpath(@__DIR__, stat_folder_name)
mkpath(stat_folder)

# File names to store the pre-processed data and retrieve them to run the programm.

training_x_name = "training_filled_x"
training_x_std_name = "training_filled_x_std"
training_y_name = "training_filled_y"

regularized_training_filled_x_name = "regularized_training_x"
regularized_training_filled_x_std_name = "regularized_training_x_std"
regularized_training_filled_x_norm_name = "regularized_training_x_norm"
regularized_training_y_name = "regularized_training_filled_y"

test_name = "test_filled"
test_std_name = "test_data_std"

regularized_test_name = "regularized_test_filled"
regularized_test_std_name = "regularized_test_data_std"
regularized_test_norm_name = "regularized_test_data_norm"

function write_preprocess_data(output_file_name, dataframe)
    serialize(joinpath(dataset_folder, output_file_name), dataframe)
end

function write_csv(output_file_name, dataframe)
    CSV.write(joinpath(output_folder, "output_" * output_file_name), dataframe)
end

function write_stat(output_file_name, dataframe)
    CSV.write(joinpath(stat_folder, "stats_" * output_file_name), dataframe)
end

Random.seed!(3)

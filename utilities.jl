using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface, Serialization

include("./save_statistics.jl")

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

dataset_folder_name = "datasets"
dataset_folder = joinpath(@__DIR__, dataset_folder_name)

training_x_name = "training_filled_x"
training_x_std_name = "training_filled_x_std"
training_y_name = "training_filled_y"

regularized_training_filled_x_name = "regularized_training_x"
regularized_training_filled_x_std_name = "regularized_training_x_std"
regularized_training_y_name = "regularized_training_filled_y"

test_name = "test_filled"
test_std_name = "test_data_std"

regularized_test_name = "regularized_test_filled"
regularized_test_std_name = "regularized_test_data_std"

function write_preprocess_data(output_file_name, dataframe)
    serialize(joinpath(dataset_folder, output_file_name), dataframe)
end

function write_csv(output_file_name, dataframe)
    CSV.write(joinpath(output_folder, output_file_name), dataframe)
end

function write_stat(output_file_name, dataframe)
    CSV.write(joinpath(stat_folder, output_file_name), dataframe)
end

Random.seed!(3)
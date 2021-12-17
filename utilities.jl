using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface

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

training_x_name = "training_filled_x.csv"
training_x_std_name = "training_filled_x_std.csv"
training_y_name = "training_filled_y.csv"

regularized_training_filled_x_name = "regularized_training_x.csv"
regularized_training_filled_x_std_name = "regularized_training_x_std.csv"
regularized_training_y_name = "regularized_training_filled_y.csv"

test_name = "test_filled.csv"
test_std_name = "test_data_std.csv"

regularized_test_name = "regularized_test_filled.csv"
regularized_test_std_name = "regularized_test_data_std.csv"

function write_preprocess_data(output_file_name, dataframe)
    CSV.write(joinpath(dataset_folder, output_file_name), dataframe)
end

function write_csv(output_file_name, dataframe)
    CSV.write(joinpath(output_folder, output_file_name), dataframe)
end

function write_stat(output_file_name, dataframe)
    CSV.write(joinpath(stat_folder, output_file_name), dataframe)
end

Random.seed!(3)
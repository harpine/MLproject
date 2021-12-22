using Pkg
Pkg.activate(@__DIR__)
using Plots, StatsPlots, DataFrames, Random, CSV, MLJ, MLJLinearModels, NearestNeighborModels, CategoricalDistributions, CategoricalArrays, MLJLIBSVMInterface, MLJDecisionTreeInterface, MLJFlux, Flux, MLJMultivariateStatsInterface, Serialization

include("./save_statistics.jl")

training_data_name = "trainingdata.csv"
test_data_name = "testdata.csv"

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
    CSV.write(joinpath(output_folder, output_file_name), dataframe)
end

function write_stat(output_file_name, dataframe)
    CSV.write(joinpath(stat_folder, output_file_name), dataframe)
end

Random.seed!(3)

machines_dictionnary = Dict("Logistic_l2" => machine_subname -> logistic_l2(machine_subname),
                        "KNN" => machine_subname -> KNN(machine_subname), 
                        "RandomForest" => machine_subname -> RandomForest(machine_subname), 
                        "Short_Neuralnetwork" => machine_subname -> short_Neuralnetwork(machine_subname), 
                        "Mlp_Neuralnetwork" => machine_subname -> mlp_Neuralnetwork(machine_subname))

"""
        *(type_machine, machine_subname, best_machines = true) # = true à mettre dans la docstring????
        Allow to choose a machine, run it, and save the different outptuts, with the subname of the machine.
        The possibilities of machine type are: "Logistic_l1" , "Logistic_l2", "KNN" "Short_Neuralnetwork", "Mlp_Neuralnetwork", "RandomForest"
"""
function run_machine(type_machine, machine_subname)
    machines_dictionnary[type_machine]

end


"""
        *(machines)
        allow to tune and apply some machines; 
        "all" = all machines
        "best" = best machines 
        "Logistic_l2", "KNN", "RandomForest", "Short_Neuralnetwork" or "Mlp_Neuralnetwork" = run only the given machine
"""
function run(machines = "1") 
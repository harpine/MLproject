include("./utilities.jl")
include("./data_preprocessing.jl")

function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--machines"
            help = "Which machine you want to run. the possibilities are: all, best, logistic_reg, knn, random_forest, short_neuralnetwork or mlp_neuralnetwork. Certain networks are very long to run!"
            arg_type = String
            required = true
            default = "best"
        "--machine_subname"
            help = "If you chose to run one specific type of machine, you can specify the string subname of the machine in the second argument. "
            arg_type = String
            default = "tuned"

    end
    return parse_args(s)
end

function preparation()

    parsed_args = parse_commandline()
    machines = parsed_args["machines"]
    machine_subname = parsed_args["machine_subname"]

    training_data = CSV.read(joinpath(@__DIR__, original_dataset_folder_name, training_data_name), DataFrame)
    test_data = CSV.read(joinpath(@__DIR__, original_dataset_folder_name, test_data_name), DataFrame)

    print("Preprocessing data: \n")
    preprocess_data(training_data, test_data)

    include("./machines.jl")
    include("./machines_run.jl")

    return machines, machine_subname
end


machines, machine_subname = preparation()
run_machines(machines, machine_subname)

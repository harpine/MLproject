# include("./data_preprocessing.jl")

# training_data = CSV.read(joinpath(@__DIR__, "datasets", training_data_name), DataFrame)
# test_data = CSV.read(joinpath(@__DIR__, "datasets", test_data_name), DataFrame)

# preprocess_data(training_data, test_data)
include("./utilities.jl")
include("./datasets")
include("./multi_logistic_reg.jl")

run_test("logistic_l2", "l2_test")

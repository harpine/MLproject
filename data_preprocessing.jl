include("./utilities.jl")
include("./sorting_data_regularization.jl")

training_data = CSV.read(joinpath(@__DIR__, "datasets", "trainingdata.csv"), DataFrame)
coerce!(training_data, :precipitation_nextday => Multiclass)

training_filled = MLJ.transform(fit!(machine(FillImputer(), training_data)), training_data)
training_filled_x = select(training_filled, Not(:precipitation_nextday))
training_filled_y = training_filled.precipitation_nextday
standardizer_mach_filled = fit!(machine(Standardizer(features = Symbol[:ALT_sunshine_4], ignore = true), training_filled_x)) #, verbosity = 2) #features ignore to retrieve too small variances
training_filled_x_std = MLJ.transform(standardizer_mach_filled, training_filled_x)

write_preprocess_data(training_x_name, training_filled_x)
write_preprocess_data(training_y_name, training_filled_y)
write_preprocess_data(training_x_std_name, training_filled_x_std)

test_data = CSV.read(joinpath(@__DIR__, "datasets", "testdata.csv"), DataFrame)
coerce!(test_data, :precipitation_nextday => Multiclass)
test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data) # We have to fill the missing datas, because we want a prediction for all existing datas.
test_data_std = MLJ.transform(standardizer_mach_filled, test_data)

write_preprocess_data(test_name, test_data)
write_preprocess_data(test_std_name, test_data_std)

sort_data_std(training_filled_x_std, training_filled_y)
sort_data_non_std(training_filled_x, training_filled_y)

training_filled_x_norm = training_filled_x
for i in 1:(size(training_filled_x_norm,2))
    if typeof(training_filled_x_norm[:, i][1]) != Float64
        training_filled_x_norm[!, i] = Float64.(training_filled_x_norm[:, i])
    end
    min = minimum(training_filled_x_norm[:, i])
    max = maximum(training_filled_x_norm[:, i])
    training_filled_x_norm[:, i] = (training_filled_x_norm[:, i] .- min) ./ (max - min)
end
training_filled_x_norm

test_data_norm = test_data
for i in 1:(size(test_data_norm,2))
    if typeof(test_data_norm[:, i][1]) != Float64
        test_data_norm[!, i] = Float64.(test_data_norm[:, i])
    end
    min = minimum(test_data_norm[:, i])
    max = maximum(test_data_norm[:, i])
    test_data_norm[:, i] = (test_data_norm[:, i] .- min) ./ (max - min)
end
test_data_norm
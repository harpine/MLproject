#include("./first_code.jl")
include("./sorting_data_regularization.jl")
include("./save_statistics.jl")

data_training_x = regularized_training_filled_x
data_training_y = training_filled_y
data_test = test_data_std
machine_subname = "CV_20_3"

model_RandomForest = RandomForestClassifier(min_samples_split = 10)
tuned_model_RandomForest = TunedModel(model = model_RandomForest,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 20),
                                range = [range(model_RandomForest, :n_trees, values = [1850, 1900, 1950]),
                                range(model_RandomForest, :max_depth , values = [95, 100, 105, 110])],
                                measure = auc)


#= #DROPPED
mach_RandomForest_d = fit!(machine(tuned_model_RandomForest,
                            training_dropped_x,
                            training_dropped_y), verbosity = 5)

# Error rate
pred_RandomForest_d = predict_mode(mach_RandomForest_d, training_dropped_x)
err_rate_RandomForest_d = mean(pred_RandomForest_d .!= training_dropped_y)
print(err_rate_RandomForest_d)

# Tuned values : n_trees, max_depth, min_samples_split
print(report(mach_RandomForest_d).best_model.n_trees)
print(report(mach_RandomForest_d).best_model.max_depth)
print(report(mach_RandomForest_d).best_model.min_samples_split) 


# Predictions
proba_RandomForest_d = predict(mach_RandomForest_d, test_data)
prediction_RandomForest_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_RandomForest_d, true))
write_csv("RandomForest_Classifier.csv", prediction_RandomForest_df) =#


# FILLED
mach_RandomForest_f = fit!(machine(tuned_model_RandomForest,
                            data_training_x,
                            data_training_y), verbosity = 4)


# Error rate
pred_RandomForest_f = predict_mode(mach_RandomForest_f, data_training_x)
err_rate_RandomForest_f = mean(pred_RandomForest_f .!= data_training_y)
print(err_rate_RandomForest_f)


# Predictions
proba_RandomForest_f = predict(mach_RandomForest_f, data_test)
prediction_RandomForest_f_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf, proba_RandomForest_f, true))
write_csv("RandomForest_Classifier_filled_" * machine_subname * ".csv", prediction_RandomForest_f_df)


# Tuned values : n_trees, max_depth, min_samples_split
print("n trees: ", report(mach_RandomForest_f).best_model.n_trees)
print("max depth: ", report(mach_RandomForest_f).best_model.max_depth)
print("min samples split: ", report(mach_RandomForest_f).best_model.min_samples_split)

MLJ.save(joinpath(machines_folder,"mach_RandomForest_filled_" * machine_subname * ".jlso"), mach_RandomForest_f)

save_statistics_randomForest(machine_subname, tuned_model_RandomForest, mach_RandomForest_f)

# test_mach = machine(joinpath(machines_folder,"mach_RandomForest_filled_server2.jlso"))
# plot(test_mach)

# Plot
plot(mach_RandomForest_f)
savefig(joinpath(plots_folder, "plot_RandomForest_tuning_auc_" * machine_subname * ".png"))

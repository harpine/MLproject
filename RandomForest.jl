include("./datasets.jl")

data_training_x = regularized_training_filled_x
data_training_y = training_filled_y
data_test = regularized_test
machine_subname = "regularized_CV_20_10"

model_RandomForest = RandomForestClassifier(#= max_depth = 40, =# min_samples_split = 35 #= , n_trees = 40 =#)
tuned_model_RandomForest = TunedModel(model = model_RandomForest,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 20),
                                range = [range(model_RandomForest, :n_trees, lower = 1500, upper = 2000),
                                range(model_RandomForest, :min_samples_split , values = [5, 10, 15]),
                                range(model_RandomForest, :max_depth , lower = 20, upper = 60)],
                                measure = auc)

mach_RandomForest = fit!(machine(tuned_model_RandomForest,
                            data_training_x,
                            data_training_y), verbosity = 4)


# Tuned values : n_trees, max_depth, min_samples_split
rep_RandomForest = report(mach_RandomForest)
print("Fitted parameters: \n")
print("n trees: ", rep_RandomForest.best_model.n_trees, "\n")
print("max depth: ", rep_RandomForest.best_model.max_depth, "\n")
print("min samples split: ", rep_RandomForest.best_model.min_samples_split, "\n", "\n")

# measurement
print("AUC measurement: ", rep_RandomForest.best_history_entry.measurement, "\n")

# Error rate
pred_RandomForest = predict_mode(mach_RandomForest, data_training_x)
err_rate_RandomForest = mean(pred_RandomForest .!= data_training_y)
print("Error rate in training set: ", err_rate_RandomForest, "\n")

# Predictions
proba_RandomForest = predict(mach_RandomForest, data_test)
prediction_RandomForest_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf, proba_RandomForest, true))
write_csv("RandomForest_Classifier_filled_" * machine_subname * ".csv", prediction_RandomForest_df)

# Plot
plot(mach_RandomForest)
savefig(joinpath(plots_folder, "plot_RandomForest_tuning_auc_" * machine_subname * ".png"))

# Saving statistics
save_statistics_randomForest(machine_subname, tuned_model_RandomForest, mach_RandomForest, data_training_x, data_training_y)

# Saving machine
MLJ.save(joinpath(machines_folder,"mach_RandomForest_filled_" * machine_subname * ".jlso"), mach_RandomForest)


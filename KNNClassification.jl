include("./sorting_data_regularization.jl")
include("./save_statistics.jl")

machine_subname = "filled_regularized_std_1"

model_KNN_class = KNNClassifier()
tuned_model_KNN_class = TunedModel(model = model_KNN_class,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = range(model_KNN_class, :K,
                                        lower = 15, upper = 25),
                                measure = auc)


# DROPPED
# mach_KNN_class_d = machine(tuned_model_KNN_class,
#                             training_dropped_x,
#                             training_dropped_y) |> fit!

# # Tuned values : K
# print(report(mach_KNN_class_d).best_model.K)

# # Error rate
# pred_KNN_class_d = predict_mode(mach_KNN_class_d, training_dropped_x)
# err_rate_KNN_class_d = mean(pred_KNN_class_d .!= training_dropped_y)
# print(err_rate_KNN_class_d)

# # Predictions
# proba_KNN_class_d = predict(mach_KNN_class_d, test_data)
# prediction_KNN_class_d_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_KNN_class_d, true))
# write_csv("KNN_classifier_dropped.csv", prediction_KNN_class_d_df)



# FILLED
mach_KNN_class_f = machine(tuned_model_KNN_class,
regularized_training_filled_x_std,
                            training_filled_y) |> fit!

# Tuned values : K
rep_KNN_class_f = report(mach_KNN_class_f)
print(rep_KNN_class_f.best_model.K)

# Error rate
pred_KNN_class_f = predict_mode(mach_KNN_class_f, regularized_training_filled_x_std)
err_rate_KNN_class_f = mean(pred_KNN_class_f .!= training_filled_y)
print(err_rate_KNN_class_f) 

# Predictions
proba_KNN_class_f = predict(mach_KNN_class_f, regularized_test_x_std)
prediction_KNN_class_f_df = DataFrame(id = 1:nrow(regularized_test_x_std), precipitation_nextday = broadcast(pdf, proba_KNN_class_f, true))
write_csv("KNN_classifier_filled_" * machine_subname * ".csv", prediction_KNN_class_f_df)

# Plot
plot(mach_KNN_class_f)
savefig(joinpath(plots_folder, "plot_KNN_class_tuning_auc_" * machine_subname * ".png"))

# Saving statistics
save_statistics_KNN_class(machine_subname, tuned_model_KNN_class, mach_KNN_class_f, true, true)

# Saving machine
MLJ.save(joinpath(machines_folder,"mach_KNN_class_filled_" * machine_subname * ".jlso"), mach_KNN_class_f)
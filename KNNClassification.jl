include("./datasets.jl")

machine_subname = "filled_regularized_std_1"

model_KNN_class = KNNClassifier()
tuned_model_KNN_class = TunedModel(model = model_KNN_class,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = range(model_KNN_class, :K,
                                        lower = 15, upper = 25),
                                measure = auc)


mach_KNN_class = machine(tuned_model_KNN_class,
regularized_training_filled_x_std,
                            training_filled_y) |> fit!

# Tuned values : K
rep_KNN_class = report(mach_KNN_class)
print(rep_KNN_class.best_model.K)

# Error rate
pred_KNN_class = predict_mode(mach_KNN_class, regularized_training_filled_x_std)
err_rate_KNN_class = mean(pred_KNN_class .!= training_filled_y)
print(err_rate_KNN_class) 

# Predictions
proba_KNN_class = predict(mach_KNN_class, regularized_test_x_std)
prediction_KNN_class_df = DataFrame(id = 1:nrow(regularized_test_x_std), precipitation_nextday = broadcast(pdf, proba_KNN_class, true))
write_csv("KNN_classifier_filled_" * machine_subname * ".csv", prediction_KNN_class_df)

# Plot
plot(mach_KNN_class)
savefig(joinpath(plots_folder, "plot_KNN_class_tuning_auc_" * machine_subname * ".png"))

# Saving statistics
save_statistics_KNN_class(machine_subname, tuned_model_KNN_class, mach_KNN_class, true, true)

# Saving machine
MLJ.save(joinpath(machines_folder,"mach_KNN_class_filled_" * machine_subname * ".jlso"), mach_KNN_class)
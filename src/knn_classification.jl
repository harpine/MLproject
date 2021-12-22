
# function knn(machine_subname)
        
#         data_training_x = regularized_training_filled_x_norm
#         data_training_y = training_filled_y
#         data_test = regularized_test_data_norm

#         model_KNN_class = KNNClassifier()
#         tuned_model_KNN_class = TunedModel(model = model_KNN_class,
#                                         tuning =  Grid(),
#                                         resampling = CV(nfolds = 20),
#                                         range = range(model_KNN_class, :K,
#                                                 lower = 23, upper = 30),
#                                         measure = auc)


#         mach_KNN_class = machine(tuned_model_KNN_class, data_training_x,
#                                 data_training_y) |> fit!

#         # Saving machine
#         MLJ.save(joinpath(machines_folder,"mach_knn_" * machine_subname * ".jlso"), mach_KNN_class)

#         # Tuned values : K
#         rep_KNN_class = report(mach_KNN_class)
#         print("Fitted parameters: \n", "K: ", rep_KNN_class.best_model.K, "\n", "\n")

#         # measurement
#         print("AUC measurement: ", rep_KNN_class.best_history_entry.measurement[1], "\n")

#         # Error rate
#         pred_KNN_class = predict_mode(mach_KNN_class, data_training_x)
#         err_rate_KNN_class = mean(pred_KNN_class .!= data_training_y)
#         print("Error rate in training set: ", err_rate_KNN_class, "\n") 

#         # AUC
#         print("AUC on training set: ", area_under_curve(predict(mach_KNN_class, data_training_x), data_training_y), "\n")

#         # Predictions
#         proba_KNN_class = predict(mach_KNN_class, data_test)
#         prediction_KNN_class_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf, proba_KNN_class, true))
#         write_csv("knn_" * machine_subname * ".csv", prediction_KNN_class_df)

#         # Plot
#         plot(mach_KNN_class)
#         savefig(joinpath(plots_folder, "plot_knn_" * machine_subname * ".png"))

#         # Saving statistics
#         save_statistics_KNN_class(machine_subname, tuned_model_KNN_class, mach_KNN_class, data_training_x, data_training_y)
# end

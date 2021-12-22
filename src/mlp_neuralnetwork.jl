
function  mlp_neuralnetwork(machine_subname)

    data_training_x = regularized_training_filled_x_std
    data_training_y = training_filled_y
    data_test = regularized_test_std

    model_Neuralnetwork = NeuralNetworkClassifier(
                                            builder = MLJFlux.Short(n_hidden = 25),
                                            optimiser = ADAMW(),
                                            lambda = 0.0,
                                            alpha = 0.0, finaliser = NNlib.softmax, batch_size = 85, epochs = 23)
    
    tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, 
                                            resampling= CV(nfolds = 10), 
                                            measure = auc, 
                                            range = [range(model_Neuralnetwork, :epochs, values = [28,30,35]),
                                                    range(model_Neuralnetwork, :(builder.hidden), values =[(100,30), (100,40)])]
                                            )

    mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, data_training_x, data_training_y), verbosity = 4)

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_Neuralnetwork_" * machine_subname * ".jlso"), mach_Neuralnetwork_tuned)

    # Tuning parameters
    print("Tuning parameters: ", tuned_model_Neuralnetwork.range, "\n", "\n") 

    # Tuned values : epochs, batch_size, lambda, alpha
    rep_Neuralnetwork = report(mach_Neuralnetwork_tuned)
    model_report = rep_Neuralnetwork.best_model
    print("Fitted parameters: \n", "epochs: ", model_report.epochs, "\n", "batch_size: ", model_report.batch_size, "\n", "lambda: ", model_report.lambda, "\n", "alpha: ", model_report.alpha, "\n", "hidden = ", model_report.builder.hidden, "\n", "\n")

    # measurement
    print("AUC measurement: ", rep_Neuralnetwork.best_history_entry.measurement, "\n")

    # Error rate
    pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, data_training_x)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= data_training_y)
    print("Error rate in training set: ", err_rate_Neuralnewtwork, "\n")

    # AUC
    print("AUC on training set:", area_under_curve(predict(mach_Neuralnetwork_tuned, data_training_x), data_training_y), "\n")

    # Predictions
    proba_Neuralnetwork = predict(mach_Neuralnetwork_tuned, data_test)
    prediction_Neuralnetwork_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf,proba_Neuralnetwork, true))
    write_csv("mlp_neuralnewtork_" * machine_subname * ".csv", prediction_Neuralnetwork_df)

    # Plot
    plot(mach_Neuralnetwork_tuned)
    savefig(joinpath(plots_folder, "plot_mlp_neuralnetwork_" * machine_subname * ".png"))

    # Saving statistics
    save_statistics_neuronal(machine_subname, tuned_model_Neuralnetwork, mach_Neuralnetwork_tuned, build_type = "mlp", regularized = true)

end
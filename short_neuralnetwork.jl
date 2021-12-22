
function short_neuralnetwork(machine_subname)

    data_training_x = regularized_training_filled_x_std
    data_training_y = training_filled_y
    data_test = regularized_test_std

    model_Neuralnetwork = NeuralNetworkClassifier(
                                            builder = MLJFlux.Short(n_hidden = 25),
                                            optimiser = ADAMW(),
                                            lambda = 0.0,
                                            alpha = 0.0, finaliser = NNlib.softmax, batch_size = 85, epochs = 23)
    
    tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, 
                                            resampling= CV(nfolds = 5), 
                                            measure = auc, 
                                            range = [range(model_Neuralnetwork, :epochs, values = [24,25,26]), 
                                                    range(model_Neuralnetwork, :batch_size, values = [80,85,90]), 
                                                    range(model_Neuralnetwork, :(builder.n_hidden), values = [25,26,27]), 
                                                    range(model_Neuralnetwork, :(builder.dropout), values = [0.4,0.5,0.6])]
                                            )

    mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, data_training_x, data_training_y), verbosity = 4)

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_short_neuralnetwork_" * machine_subname * ".jlso"), mach_Neuralnetwork_tuned)

    # Tuning parameters
    print("Tuning parameters: ", tuned_model_Neuralnetwork.range, "\n", "\n") 

    # Tuned values : epochs, batch_size, lambda, alpha, n_hidden, dropout
    rep_Neuralnetwork = report(mach_Neuralnetwork_tuned)
    model_report = rep_Neuralnetwork.best_model
    print("Fitted parameters: \n", "epochs: ", model_report.epochs, "\n", "batch_size: ", model_report.batch_size, "\n", "lambda: ", model_report.lambda, "\n", "alpha: ", model_report.alpha, "\n", "n_hidden: ", model_report.builder.n_hidden, "\n", "dropout: ", model_report.builder.dropout, "\n", "\n")

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
    write_csv("short_neuralnetwork_" * machine_subname * ".csv", prediction_Neuralnetwork_df)

    # Plot
    plot(mach_Neuralnetwork_tuned)
    savefig(joinpath(plots_folder, "plot_short_neuralnetwork_" * machine_subname * ".png"))

    # Saving statistics
    save_statistics_neuronal(machine_subname, tuned_model_Neuralnetwork, mach_Neuralnetwork_tuned, build_type = "short", regularized = true)
end
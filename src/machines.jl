# Multiple logistic regression with ridge regularization (l2) tuned:

function logistic_reg_l2(machine_subname)

    data_training_x = training_filled_x_std
    data_training_y = training_filled_y
    data_test = test_data_std

    model_mult_logistic_reg_l2 = LogisticClassifier(penalty = :l2)

    tuned_model_mult_logistic_reg_l2 = TunedModel(model = model_mult_logistic_reg_l2,
                                    resampling = CV(nfolds = 10),
                                    tuning = Grid(),
                                    range = range(model_mult_logistic_reg_l2, :lambda, lower = 50 , upper = 200, scale = :log),
                                    measure = auc)


    mach_mult_logistic_reg_l2 = fit!(machine(tuned_model_mult_logistic_reg_l2, data_training_x, data_training_y))

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_logistic_reg_l2_" * machine_subname * ".jlso"), mach_mult_logistic_reg_l2)

    # Tuned values : lambda
    rep_logistic_reg = report(mach_mult_logistic_reg_l2)
    print("Fitted parameters: \n")
    print("lambda: ", rep_logistic_reg.best_model.lambda, "\n", "\n")

    # measurement
    print("AUC measurement: ", rep_logistic_reg.best_history_entry.measurement[1], "\n")

    # Error rate
    pred_mult_logistic_reg_l2 = predict_mode(mach_mult_logistic_reg_l2, data_training_x)
    err_rate_mult_logistic_reg_l2 = mean(pred_mult_logistic_reg_l2 .!= data_training_y)
    print("Error rate in training set: ", err_rate_mult_logistic_reg_l2, "\n")

    # AUC
    print("AUC on training set: ", area_under_curve(predict(mach_mult_logistic_reg_l2, data_training_x), data_training_y), "\n")

    # Predictions
    proba_mult_logistic_reg_l2 = broadcast(pdf, predict(mach_mult_logistic_reg_l2, data_test), true)
    prediction_mult_logistic_reg_l2_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = proba_mult_logistic_reg_l2)
    write_csv("logistic_reg_l2_" * machine_subname * ".csv", prediction_mult_logistic_reg_l2_df)

    # Plot
    plot(mach_mult_logistic_reg_l2)
    savefig(joinpath(plots_folder, "plot_logistic_reg_l2_" * machine_subname * ".png"))

    # Saving statistics
    save_logistic_reg(machine_subname, tuned_model_mult_logistic_reg_l2, mach_mult_logistic_reg_l2)

end


function knn(machine_subname)
        
    data_training_x = regularized_training_filled_x_norm
    data_training_y = training_filled_y
    data_test = regularized_test_data_norm

    model_KNN_class = KNNClassifier()
    tuned_model_KNN_class = TunedModel(model = model_KNN_class,
                                    tuning =  Grid(),
                                    resampling = CV(nfolds = 20),
                                    range = range(model_KNN_class, :K,
                                            lower = 23, upper = 30),
                                    measure = auc)


    mach_KNN_class = machine(tuned_model_KNN_class, data_training_x,
                            data_training_y) |> fit!

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_knn_" * machine_subname * ".jlso"), mach_KNN_class)

    # Tuned values : K
    rep_KNN_class = report(mach_KNN_class)
    print("Fitted parameters: \n", "K: ", rep_KNN_class.best_model.K, "\n", "\n")

    # measurement
    print("AUC measurement: ", rep_KNN_class.best_history_entry.measurement[1], "\n")

    # Error rate
    pred_KNN_class = predict_mode(mach_KNN_class, data_training_x)
    err_rate_KNN_class = mean(pred_KNN_class .!= data_training_y)
    print("Error rate in training set: ", err_rate_KNN_class, "\n") 

    # AUC
    print("AUC on training set: ", area_under_curve(predict(mach_KNN_class, data_training_x), data_training_y), "\n")

    # Predictions
    proba_KNN_class = predict(mach_KNN_class, data_test)
    prediction_KNN_class_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf, proba_KNN_class, true))
    write_csv("knn_" * machine_subname * ".csv", prediction_KNN_class_df)

    # Plot
    plot(mach_KNN_class)
    savefig(joinpath(plots_folder, "plot_knn_" * machine_subname * ".png"))

    # Saving statistics
    save_statistics_KNN_class(machine_subname, tuned_model_KNN_class, mach_KNN_class, data_training_x, data_training_y)
end


function random_forest(machine_subname)
    data_training_x = training_filled_x
    data_training_y = training_filled_y
    data_test = test_data

    model_RandomForest = RandomForestClassifier(min_samples_split = 10)
    tuned_model_RandomForest = TunedModel(model = model_RandomForest,
                                    tuning =  Grid(),
                                    resampling = CV(nfolds = 20),
                                    range = [range(model_RandomForest, :n_trees, values = [1859, 1900, 1950]),
                                    range(model_RandomForest, :max_depth , values = [95, 100, 105, 110])],
                                    measure = auc)

    mach_RandomForest = fit!(machine(tuned_model_RandomForest,
                                data_training_x,
                                data_training_y), verbosity = 4)

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_random_forest_" * machine_subname * ".jlso"), mach_RandomForest)

    # Tuned values : n_trees, max_depth, min_samples_split
    rep_RandomForest = report(mach_RandomForest)
    print("Fitted parameters: \n")
    print("n trees: ", rep_RandomForest.best_model.n_trees, "\n")
    print("max depth: ", rep_RandomForest.best_model.max_depth, "\n")
    print("min samples split: ", rep_RandomForest.best_model.min_samples_split, "\n", "\n")

    # measurement
    print("AUC measurement: ", rep_RandomForest.best_history_entry.measurement[1], "\n")

    # Error rate
    pred_RandomForest = predict_mode(mach_RandomForest, data_training_x)
    err_rate_RandomForest = mean(pred_RandomForest .!= data_training_y)
    print("Error rate in training set: ", err_rate_RandomForest, "\n")

    # AUC
    print("AUC on training set: ", area_under_curve(predict(mach_RandomForest, data_training_x), data_training_y), "\n")

    # Predictions
    proba_RandomForest = predict(mach_RandomForest, data_test)
    prediction_RandomForest_df = DataFrame(id = 1:nrow(data_test), precipitation_nextday = broadcast(pdf, proba_RandomForest, true))
    write_csv("random_forest_" * machine_subname * ".csv", prediction_RandomForest_df)

    # Saving statistics
    save_statistics_randomForest(machine_subname, tuned_model_RandomForest, mach_RandomForest, data_training_x, data_training_y)

    # Plot
    plot(mach_RandomForest)
    savefig(joinpath(plots_folder, "plot_random_forest_" * machine_subname * ".png"))

end


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
                                            range = [range(model_Neuralnetwork, :epochs, values = [20,21,22,23,24,25,26,27,28]), range(model_Neuralnetwork, :batch_size, values = [80,85,90,95]), range(model_Neuralnetwork, :(builder.n_hidden), values = [23,24,25,26,27]), range(model_Neuralnetwork, :(builder.dropout), values = [0.0,0.2,0.5,0.7])]#, acceleration=CUDALibs()) #, tune: optimiser, 
                                            # range = [range(model_Neuralnetwork, :epochs, values = [24,25,26]), 
                                            #         range(model_Neuralnetwork, :batch_size, values = [80,85,90]), 
                                            #         range(model_Neuralnetwork, :(builder.n_hidden), values = [25,26,27]), 
                                            #         range(model_Neuralnetwork, :(builder.dropout), values = [0.4,0.5,0.6])]
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
    print("AUC measurement: ", rep_Neuralnetwork.best_history_entry.measurement[1], "\n")

    # Error rate
    pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, data_training_x)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= data_training_y)
    print("Error rate in training set: ", err_rate_Neuralnewtwork, "\n")

    # AUC
    print("AUC on training set: ", area_under_curve(predict(mach_Neuralnetwork_tuned, data_training_x), data_training_y), "\n")

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


function  mlp_neuralnetwork(machine_subname)

    data_training_x = regularized_training_filled_x_std
    data_training_y = training_filled_y
    data_test = regularized_test_std

    model_Neuralnetwork = NeuralNetworkClassifier(
                                            builder = MLJFlux.MLP(hidden=(100,30)),
                                            optimiser = ADAMW(),
                                            lambda = 0.0,
                                            alpha = 0.0, batch_size = 32, finaliser = NNlib.softmax, epochs = 15)
    
    
    tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, 
                                            resampling= CV(nfolds = 20), 
                                            measure = auc, 
                                            range = [range(model_Neuralnetwork, :batch_size, values =[30,31,32])]
                                            #[range(model_Neuralnetwork, :epochs, values = [28,30,35]),
                                                    #range(model_Neuralnetwork, :(builder.hidden), values =[(100,30), (100,40)])]
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
    print("AUC measurement: ", rep_Neuralnetwork.best_history_entry.measurement[1], "\n")

    # Error rate
    pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, data_training_x)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= data_training_y)
    print("Error rate in training set: ", err_rate_Neuralnewtwork, "\n")

    # AUC
    print("AUC on training set: ", area_under_curve(predict(mach_Neuralnetwork_tuned, data_training_x), data_training_y), "\n")

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
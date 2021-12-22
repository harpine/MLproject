include("./datasets.jl")


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
    print("AUC measurement: ", rep_RandomForest.best_history_entry.measurement, "\n")

    # Error rate
    pred_RandomForest = predict_mode(mach_RandomForest, data_training_x)
    err_rate_RandomForest = mean(pred_RandomForest .!= data_training_y)
    print("Error rate in training set: ", err_rate_RandomForest, "\n")

    # AUC
    print("AUC on training set:", area_under_curve(predict(mach_RandomForest, data_training_x), data_training_y), "\n")

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

random_forest("test_CV20_3")

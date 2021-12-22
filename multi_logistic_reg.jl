include("./datasets.jl")

# Multiple logistic regression with ridge regularization (l2) tuned:

function logistic_l2(machine_subname)
    model_mult_logistic_reg_l2 = LogisticClassifier(penalty = :l2)

    tuned_model_mult_logistic_reg_l2 = TunedModel(model = model_mult_logistic_reg_l2,
                                    resampling = CV(nfolds = 10),
                                    tuning = Grid(),
                                    range = range(model_mult_logistic_reg_l2, :lambda, lower = 50 , upper = 200, scale = :log),
                                    measure = auc)


    mach_mult_logistic_reg_l2 = fit!(machine(tuned_model_mult_logistic_reg_l2, training_filled_x_std, training_filled_y))

    # Tuned values : lambda
    rep_logistic_reg = report(mach_mult_logistic_reg_l2)
    print("Fitted parameters: \n")
    print("lambda: ", rep_logistic_reg.best_model.lambda, "\n", "\n")

    # measurement
    print("AUC measurement: ", rep_logistic_reg.best_history_entry.measurement, "\n")

    # Error rate
    pred_mult_logistic_reg_l2 = predict_mode(mach_mult_logistic_reg_l2, training_filled_x_std)
    err_rate_mult_logistic_reg_l2 = mean(pred_mult_logistic_reg_l2 .!= training_filled_y)
    print("Error rate in training set: ", err_rate_mult_logistic_reg_l2, "\n")

    # AUC
    print("AUC on training set:", area_under_curve(predict(mach_mult_logistic_reg_l2, training_filled_x_std), training_filled_y), "\n")

    # Predictions
    proba_mult_logistic_reg_l2 = broadcast(pdf, predict(mach_mult_logistic_reg_l2, test_data_std), true)
    prediction_mult_logistic_reg_l2_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_l2)
    write_csv("logistic_reg_l2_" * machine_subname * ".csv", prediction_mult_logistic_reg_l2_df)

    # Plot
    plot(mach_mult_logistic_reg_l2)
    savefig(joinpath(plots_folder, "plot_logistic_reg_l2_" * machine_subname * ".png"))

    # Saving statistics
    save_logistic_reg(machine_subname, tuned_model_mult_logistic_reg_l2, mach_mult_logistic_reg_l2)

    # Saving machine
    MLJ.save(joinpath(machines_folder,"mach_logistic_reg_l2_" * machine_subname * ".jlso"), mach_mult_logistic_reg_l2)

end

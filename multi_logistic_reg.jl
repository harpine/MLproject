include("./datasets.jl")

# Multiple logistic regression with lasso regularization (l1) tuned:

machine_subname = "l1_1"

model_mult_logistic_reg_l1 = LogisticClassifier(penalty = :l1, solver = ISTA(tol = 5e-4)) # solver to avoid warning : Proximal GD did not converge in 1000 iterations.

tuned_model_mult_logistic_reg_l1 = TunedModel(model = model_mult_logistic_reg_l1,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model_mult_logistic_reg_l1, :lambda, lower = 1 , upper = 10, scale = :log),
                                   measure = auc)

mach_mult_logistic_reg_l1 = fit!(machine(tuned_model_mult_logistic_reg_l1, training_filled_x_std, training_filled_y))
pred_mult_logistic_reg_l1 = predict_mode(mach_mult_logistic_reg_l1, training_filled_x_std)
err_rate_mult_logistic_reg_l1 = mean(pred_mult_logistic_reg_l1 .!= training_filled_y)

plot(mach_mult_logistic_reg_l1)
savefig(joinpath(plots_folder, "machine_plot" * machine_subname * "png"))

print("lambda : ", report(mach_mult_logistic_reg_l1).best_history_entry.model.lambda, "\n")
print("l1 : ", err_rate_mult_logistic_reg_l1, "\n")

proba_mult_logistic_reg_l1 = broadcast(pdf, predict(mach_mult_logistic_reg_l1, test_data_std), true)
prediction_mult_logistic_reg_l1_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_l1)
write_csv("multi_logistic_reg_tuned_l1_filled_" * machine_subname * ".csv", prediction_mult_logistic_reg_l1_df)

save_logistic_reg(machine_subname, tuned_model_mult_logistic_reg_l1,mach_mult_logistic_reg_l1)


# Multiple logistic regression with ridge regularization (l2) tuned:

machine_subname = "l2_1"
model_mult_logistic_reg_l2 = LogisticClassifier(penalty = :l2)

tuned_model_mult_logistic_reg_l2 = TunedModel(model = model_mult_logistic_reg_l2,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model_mult_logistic_reg_l2, :lambda, lower = 50 , upper = 200, scale = :log),
                                   measure = auc)


mach_mult_logistic_reg_l2 = fit!(machine(tuned_model_mult_logistic_reg_l2, training_filled_x_std, training_filled_y))

print("lambda : ", report(mach_mult_logistic_reg_l2).best_history_entry.model.lambda, "\n")

pred_mult_logistic_reg_l2 = predict_mode(mach_mult_logistic_reg_l2, training_filled_x_std)
err_rate_mult_logistic_reg_l2 = mean(pred_mult_logistic_reg_l2 .!= training_filled_y)

print("l2 : ", err_rate_mult_logistic_reg_l2, "\n")

proba_mult_logistic_reg_l2 = broadcast(pdf, predict(mach_mult_logistic_reg_l2, test_data_std), true)
prediction_mult_logistic_reg_l2_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_l2)
write_csv("multi_logistic_reg_tuned_l2_filled_" * machine_subname * ".csv", prediction_mult_logistic_reg_l2_df)

save_logistic_reg(machine_subname, tuned_model_mult_logistic_reg_l2, mach_mult_logistic_reg_l2)

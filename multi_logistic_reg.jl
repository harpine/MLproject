include("./first_code.jl")

model_not_reg = LogisticClassifier(penalty = :none)
mach_mult_logistic_not_reg = fit!(machine(model_not_reg, training_dropped_x,
training_dropped_y))
pred_mult_logistic_not_reg = predict_mode(mach_mult_logistic_not_reg, training_dropped_x)
err_rate_mult_logistic_not_reg = mean(pred_mult_logistic_not_reg .!= training_dropped_y)

print("no regularization : ", err_rate_mult_logistic_not_reg, "\n")

proba_mult_logistic_not_reg = predict(mach_mult_logistic_not_reg, test_data)
prediction_mult_logistic_not_reg_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf,proba_mult_logistic_not_reg, true))
write_csv("multi_logistic_reg_not_reg.csv", prediction_mult_logistic_not_reg_df)

# model_not_tuned_l2 = LogisticClassifier(penalty = :l2, lambda = 2e-6)
# mach_mult_logistic_not_tuned_l2 = fit!(machine(model_not_tuned_l2, training_dropped_x,
# training_dropped_y))
# pred_mult_logistic_not_tuned_l2 = predict_mode(mach_mult_logistic_not_tuned_l2, training_dropped_x)
# err_rate_mult_logistic_not_tuned_l2 = mean(pred_mult_logistic_not_tuned_l2 .!= training_dropped_y)
# print("ridge not tuned : ", err_rate_mult_logistic_not_tuned_l2)

# proba_mult_logistic_not_tuned_l2 = predict(mach_mult_logistic_not_tuned_l2, test_data)
# prediction_mult_logistic_not_tuned_l2_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf,proba_mult_logistic_not_tuned_l2, true))
# write_csv("multi_logistic_reg_not_tuned.csv", prediction_mult_logistic_not_tuned_l2_df)


# model_not_tuned_1 = LogisticClassifier(penalty = :l1, lambda = 1e-4, solver = ISTA(tol = 1e-4)) # default tolerance = 1e-4
# mach_mult_logistic_not_tuned_1 = fit!(machine(model_not_tuned_1, training_dropped_x,
# training_dropped_y))
# pred_mult_logistic_not_tuned_1 = predict_mode(mach_mult_logistic_not_tuned_1, training_dropped_x)
# #proba_mult_logistic_reg_d_l1 = eval(predict(mach_mult_logistic_reg_d_l1, training_dropped_x_std), training_dropped_y)
# err_rate_mult_logistic_not_tuned_1 = mean(pred_mult_logistic_not_tuned_1 .!= training_dropped_y)

# print("lasso not tuned : ", err_rate_mult_logistic_not_tuned_1)



# model_mult_logistic_reg_l1 = LogisticClassifier(penalty = :l1, solver = ISTA(tol = 5e-4)) # solver to avoid warning : Proximal GD did not converge in 1000 iterations.

# tuned_model_mult_logistic_reg_l1 = TunedModel(model = model_mult_logistic_reg_l1,
#                                    resampling = CV(nfolds = 10),
#                                    tuning = Grid(),
#                                    range = range(model_mult_logistic_reg_l1, :lambda, lower = 6 , upper = 8, scale = :log),
#                                    measure = auc)

                              
# mach_mult_logistic_reg_d_l1 = fit!(machine(tuned_model_mult_logistic_reg_l1, training_dropped_x_std, training_dropped_y))
# pred_mult_logistic_reg_d_l1 = predict_mode(mach_mult_logistic_reg_d_l1, training_dropped_x_std)
# err_rate_mult_logistic_reg_d_l1 = mean(pred_mult_logistic_reg_d_l1 .!= training_dropped_y)

# print("lambda : ", report(mach_mult_logistic_reg_d_l1).best_history_entry.model.lambda, "\n")
# print("l1 : ", err_rate_mult_logistic_reg_d_l1, "\n")

# proba_mult_logistic_reg_d_l1 = broadcast(pdf, predict(mach_mult_logistic_reg_d_l1, test_data_std), true)
# prediction_mult_logistic_reg_d_l1_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_d_l1)
# write_csv("multi_logistic_reg_tuned_l1.csv", prediction_mult_logistic_reg_d_l1_df)



model_mult_logistic_reg_l2 = LogisticClassifier(penalty = :l2)

tuned_model_mult_logistic_reg_l2 = TunedModel(model = model_mult_logistic_reg_l2,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   #range = range(model_mult_logistic_reg_l2, :lambda, lower = 3e-2, upper = 3e2, scale = :log),
                                   range = range(model_mult_logistic_reg_l2, :lambda, lower = 200 , upper = 250, scale = :log),
                                   measure = auc)


# mach_mult_logistic_reg_d_l2 = fit!(machine(tuned_model_mult_logistic_reg_l2, training_dropped_x_std, training_dropped_y))

# print("lambda : ", report(mach_mult_logistic_reg_d_l2).best_history_entry.model.lambda, "\n")

# pred_mult_logistic_reg_d_l2 = predict_mode(mach_mult_logistic_reg_d_l2, training_dropped_x_std)
# err_rate_mult_logistic_reg_d_l2 = mean(pred_mult_logistic_reg_d_l2 .!= training_dropped_y)

# print("l2 : ", err_rate_mult_logistic_reg_d_l2, "\n")

# proba_mult_logistic_reg_d_l2 = broadcast(pdf, predict(mach_mult_logistic_reg_d_l2, test_data_std), true)
# prediction_mult_logistic_reg_d_l2_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_d_l2)
# write_csv("multi_logistic_reg_tuned_l2_.csv", prediction_mult_logistic_reg_d_l2_df)

mach_mult_logistic_reg_f_l2 = fit!(machine(tuned_model_mult_logistic_reg_l2, training_filled_x_std, training_filled_y))

print("lambda : ", report(mach_mult_logistic_reg_f_l2).best_history_entry.model.lambda, "\n")

pred_mult_logistic_reg_f_l2 = predict_mode(mach_mult_logistic_reg_f_l2, training_filled_x_std)
err_rate_mult_logistic_reg_f_l2 = mean(pred_mult_logistic_reg_f_l2 .!= training_filled_y)

print("l2 : ", err_rate_mult_logistic_reg_f_l2, "\n")

proba_mult_logistic_reg_f_l2 = broadcast(pdf, predict(mach_mult_logistic_reg_f_l2, test_data_std), true)
prediction_mult_logistic_reg_f_l2_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = proba_mult_logistic_reg_f_l2)
write_csv("multi_logistic_reg_tuned_l2_filled.csv", prediction_mult_logistic_reg_f_l2_df)

# proba_mult_logistic_reg_d_l2_train = broadcast(pdf, predict(mach_mult_logistic_reg_d_l2, training_dropped_x_std), true)
# print(typeof(proba_mult_logistic_reg_d_l2_train))
# print(typeof(training_dropped_y))
# proba_training_dropped_y = coerce(training_dropped_y, Count) .-1
# print(rmse(proba_mult_logistic_reg_d_l2_train, proba_training_dropped_y))
# prediction_mult_logistic_reg_d_l2_df_train = DataFrame(id = 1:nrow(training_dropped_x_std), precipitation_nextday = proba_mult_logistic_reg_d_l2_train)
# write_csv("multi_logistic_reg_tuned_l2_train.csv", prediction_mult_logistic_reg_d_l2_df_train)


include("./first_code.jl")

model_not_tuned = LogisticClassifier(penalty = :none)
mach_mult_logistic_reg_d_l = fit!(machine(model_not_tuned, training_dropped_x,
training_dropped_y))
pred_mult_logistic_reg_d_l = predict_mode(mach_mult_logistic_reg_d_l, training_dropped_x)
#proba_mult_logistic_reg_d_l1 = eval(predict(mach_mult_logistic_reg_d_l1, training_dropped_x), training_dropped_y)
err_rate_mult_logistic_reg_d_l = mean(pred_mult_logistic_reg_d_l .!= training_dropped_y)

print("no tuning : ", err_rate_mult_logistic_reg_d_l)



model_mult_logistic_reg_l1 = LogisticClassifier(penalty = :l1)

tuned_model_mult_logistic_reg_l1 = TunedModel(model = model_mult_logistic_reg_l1,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model_mult_logistic_reg_l1, :lambda, lower = 1e-30 , upper = 1e-20, scale = :log),
                                   #range = range(model_mult_logistic_reg_l1, :lambda, lower = 1e-15 / (2 * size(training_dropped_x)[1]), upper = 1e-10 / (2 * size(training_dropped_x)[1]), scale = :log),
                                   measure = auc)

                              
mach_mult_logistic_reg_d_l1 = fit!(machine(tuned_model_mult_logistic_reg_l1, training_dropped_x,
training_dropped_y))
pred_mult_logistic_reg_d_l1 = predict_mode(mach_mult_logistic_reg_d_l1, training_dropped_x)
#proba_mult_logistic_reg_d_l1 = eval(predict(mach_mult_logistic_reg_d_l1, training_dropped_x), training_dropped_y)
err_rate_mult_logistic_reg_d_l1 = mean(pred_mult_logistic_reg_d_l1 .!= training_dropped_y)

print("l1 : ", err_rate_mult_logistic_reg_d_l1)

model_mult_logistic_reg_l2 = LogisticClassifier()

tuned_model_mult_logistic_reg_l2 = TunedModel(model = model_mult_logistic_reg_l2,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model_mult_logistic_reg_l1, :lambda, lower = 1e-30 , upper = 1e-20, scale = :log),
                                   #range = range(model_mult_logistic_reg_l2, :lambda, lower = 1e-15 / (2 * size(training_dropped_x)[1]), upper = 1e-5 / (2 * size(training_dropped_x)[1]), scale = :log),
                                   measure = auc)

                                
mach_mult_logistic_reg_d_l2 = fit!(machine(tuned_model_mult_logistic_reg_l1, training_dropped_x,
training_dropped_y))
pred_mult_logistic_reg_d_l2 = predict_mode(mach_mult_logistic_reg_d_l2, training_dropped_x)
#proba_mult_logistic_reg_d_l2 = eval(predict(mach_mult_logistic_reg_d_l2, training_dropped_x), training_dropped_y)
err_rate_mult_logistic_reg_d_l2 = mean(pred_mult_logistic_reg_d_l2 .!= training_dropped_y)

print("l2 : ", err_rate_mult_logistic_reg_d_l2)

#mach_mult_logistic_reg_m
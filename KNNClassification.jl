include("./first_code.jl")

model_KNN_class = KNNClassifier()
tuned_model_KNN_class = TunedModel(model = model_KNN_class,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = range(model_KNN_class, :K,
                                        lower = 25, upper = 34),
                                measure = auc)


# DROPPED
# mach_KNN_class_d = machine(tuned_model_KNN_class,
#                             training_dropped_x,
#                             training_dropped_y) |> fit!

# # Tuned values : K
# print(report(mach_KNN_class_d).best_model.K)

# # Error rate
# pred_KNN_class_d = predict_mode(mach_KNN_class_d, training_dropped_x)
# err_rate_KNN_class_d = mean(pred_KNN_class_d .!= training_dropped_y)
# print(err_rate_KNN_class_d)

# # Predictions
# proba_KNN_class_d = predict(mach_KNN_class_d, test_data)
# prediction_KNN_class_d_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_KNN_class_d, true))
# write_csv("KNN_classifier_dropped.csv", prediction_KNN_class_d_df)



# FILLED
mach_KNN_class_f = machine(tuned_model_KNN_class,
                            training_filled_x,
                            training_filled_y) |> fit!

# Tuned values : K
print(report(mach_KNN_class_f).best_model.K)

# Error rate
pred_KNN_class_f = predict_mode(mach_KNN_class_f, training_filled_x)
err_rate_KNN_class_f = mean(pred_KNN_class_f .!= training_filled_y)
print(err_rate_KNN_class_f)

# Predictions
proba_KNN_class_f = predict(mach_KNN_class_f, test_data)
prediction_KNN_class_f_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_KNN_class_f, true))
write_csv("KNN_classifier_filled.csv", prediction_KNN_class_f_df)
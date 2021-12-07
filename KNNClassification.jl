include("./first_code.jl")

model_KNN_class = KNNClassifier()
tuned_model_KNN_class = TunedModel(model = model_KNN_class,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = range(model_KNN_class, :K,
                                        lower = 1, upper = 50),
                                measure = auc)
mach_KNN_class_d = machine(tuned_model_KNN_class,
                            training_dropped_x,
                            training_dropped_y) |> fit!

# rep = report(mach_KNN_class_d)

# bbest = rep.best_model.K

pred_KNN_class_d = predict_mode(mach_KNN_class_d, training_dropped_x)

err_rate_KNN_class_d = mean(pred_KNN_class_d .!= training_dropped_y)

print(err_rate_KNN_class_d)


proba_KNN_class_d = predict(mach_KNN_class_d, test_data)
prediction_KNN_class_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_KNN_class_d, true))
write_csv("KNN_classifier.csv", prediction_KNN_class_df)

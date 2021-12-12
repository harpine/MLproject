include("./first_code.jl")

model_RandomForest = RandomForestClassifier(n_trees = 1367, max_depth = 40)
tuned_model_RandomForest = TunedModel(model = model_RandomForest,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = range(model_RandomForest, :min_samples_split, lower = 2, upper = 80),
                                measure = auc)
mach_RandomForest_d = fit!(machine(tuned_model_RandomForest,
                            training_dropped_x,
                            training_dropped_y), verbosity = 5)

# Error rate
pred_RandomForest_d = predict_mode(mach_RandomForest_d, training_dropped_x)
err_rate_RandomForest_d = mean(pred_RandomForest_d .!= training_dropped_y)
print(err_rate_RandomForest_d)

# Tuned values : n_trees, max_depth, min_samples_split
print(report(mach_RandomForest_d).best_model.n_trees)
print(report(mach_RandomForest_d).best_model.max_depth)
print(report(mach_RandomForest_d).best_model.min_samples_split)

# Predictions
proba_RandomForest_d = predict(mach_RandomForest_d, test_data)
prediction_RandomForest_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_RandomForest_d, true))
write_csv("RandomForest_Classifier.csv", prediction_RandomForest_df)



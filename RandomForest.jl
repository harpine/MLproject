include("./first_code.jl")

model_RandomForest = RandomForestClassifier()
tuned_model_RandomForest = TunedModel(model = model_RandomForest,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 10),
                                range = [range(model_RandomForest, :n_trees, lower = 1000, upper = 2000),
                                range(model_RandomForest, :min_samples_split, lower = 20, upper = 60),
                                range(model_RandomForest, :max_depth , lower = 20, upper = 60)],
                                measure = auc)


#= #DROPPED
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
write_csv("RandomForest_Classifier.csv", prediction_RandomForest_df) =#


# FILLED
mach_RandomForest_f = fit!(machine(tuned_model_RandomForest,
                            training_filled_x,
                            training_filled_y), verbosity = 5)

MLJ.save(joinpath(machines_folder,"mach_RandomForest_filled_server1.jlso"), mach_Neuralnetwork_tuned)
# Error rate
pred_RandomForest_f = predict_mode(mach_RandomForest_f, training_filled_x)
err_rate_RandomForest_f = mean(pred_RandomForest_f .!= training_filled_y)
print(err_rate_RandomForest_f)

# Tuned values : n_trees, max_depth, min_samples_split
print(report(mach_RandomForest_f).best_model.n_trees)

# Predictions
proba_RandomForest_f = predict(mach_RandomForest_f, test_data)
prediction_RandomForest_f_df = DataFrame(id = 1:nrow(test_data), precipitation_nextday = broadcast(pdf, proba_RandomForest_f, true))
write_csv("RandomForest_Classifier_filled.csv", prediction_RandomForest_f_df)

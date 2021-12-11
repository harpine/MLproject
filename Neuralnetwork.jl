include("./first_code.jl")

# builder = MLJFlux.Short(n_hidden = 128, σ = relu)
#optimiser = ADAM()

#https://github.com/FluxML/MLJFlux.jl

# model_Neuralnetwork = @pipeline( Standardizer(features = Symbol[:ALT_sunshine_4, :ZER_sunshine_1], ignore = true), NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
#                                                                                                         Dense(100, n_out, sigmoid))),
#                                                                                                         optimiser = ADAMW(),
#                                                                                                     batch_size = 32,epochs = 20), 
#                                                                                                     target = Standardizer(), prediction_type = :probabilistic)

model_Neuralnetwork = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                                                                Dense(100,100,relu),
                                                                                Dense(100, n_out, sigmoid))),
                                                                                optimiser = ADAMW(),
                                                                            batch_size = 32, epochs = 20)

#model_Neuralnetwork = @pipeline(Standardizer(), NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, σ = sigmoid, dropout = 0.5), optimiser = ADAMW()), target = Standardizer())

# model_Neuralnetwork = NeuralNetworkClassifier(
# 	builder = Short(
# 			n_hidden = 0,
# 			dropout = 0.5,
# 			σ = NNlib.σ),
# 	finaliser = NNlib.softmax,
# 	optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}()),
# 	loss = Flux.crossentropy,
# 	epochs = 10,
# 	batch_size = 1,
# 	lambda = 0.0,
# 	alpha = 0.0,
# 	optimiser_changes_trigger_retraining = false)

tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 5), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5, 20, 100, 500, 1000]), range(model_Neuralnetwork, :lambda, lower = 2e-4 , upper = 2e4, scale = :log)])

mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, training_dropped_x_std, training_dropped_y), verbosity = 8)

#fitted_params(tuned_model_Neuralnetwork)
#report(mach_Neuralnetwork_tuned).fitted_params
pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, training_dropped_x_std)

err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_dropped_y)

print("Neural Network: ", err_rate_Neuralnewtwork, "\n")

proba_Neuralnetwork = predict(mach_Neuralnetwork_tuned, test_data_std)
prediction_Neuralnetwork_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = broadcast(pdf,proba_Neuralnetwork, true))
write_csv("neural_newtork.csv", prediction_Neuralnetwork_df)


# dans le rapport: pourquoi avoir choisi quelle méthode, expliquer où on a passé du temps pourquoi et comment on a résolu 
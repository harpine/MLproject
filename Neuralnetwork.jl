include("./first_code.jl")

machines_folder = "machines"
losses_folder = "losses"
mkpath(machines_folder)
mkpath(losses_folder)


# builder = MLJFlux.Short(n_hidden = 128, σ = relu)
#optimiser = ADAM()

#https://github.com/FluxML/MLJFlux.jl

# model_Neuralnetwork = @pipeline( Standardizer(features = Symbol[:ALT_sunshine_4, :ZER_sunshine_1], ignore = true), NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
#                                                                                                         Dense(100, n_out, sigmoid))),
#                                                                                                         optimiser = ADAMW(),
#                                                                                                     batch_size = 32,epochs = 20), 
#                                                                                                     target = Standardizer(), prediction_type = :probabilistic)

# A noter dans le rapport: The sigmoid function is used for the two-class logistic regression, whereas the softmax function is used for the multiclass logistic regression (a.k.a. MaxEnt, multinomial logistic regression, softmax Regression, Maximum Entropy Classifier).


# losses = []
# training_losses = []
# parameter_means = Float32[];
# epochs = []

model_Neuralnetwork = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                                                                #Dense(100,100,relu),
                                                                                #Dense(100,100,relu),
                                                                                Dense(100, n_out, sigmoid))),
                                                                                optimiser = ADAMW(), batch_size = 128) # , finaliser = NNlib.sigmoid: I can't implement it!!


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

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 5), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5, 20, 100, 500, 1000]), range(model_Neuralnetwork, :lambda, lower = 2e-4 , upper = 2e4, scale = :log)]) #acceleration=CUDALibs(), tune: optimiser, 
# submission 3 

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5,250,500]), range(model_Neuralnetwork, :lambda, lower = 2e-3 , upper = 2e-1, scale = :log), range(model_Neuralnetwork, :alpha, values = [0,0.5,1] )])#, acceleration=CUDALibs()) #, tune: optimiser, 
#server1

tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5,10,20]), range(model_Neuralnetwork, :lambda, lower = 2e-5 , upper = 2e-2, scale = :log), range(model_Neuralnetwork, :alpha, values = [0,0.5,1.0] )])#, acceleration=CUDALibs()) #, tune: optimiser, 
#server2

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 5), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5, 10, 15]), range(model_Neuralnetwork, :lambda, lower = 2e-1 , upper = 2, scale = :log)])#, acceleration=CUDALibs()) #, tune: optimiser, 
#s5-test


mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, training_dropped_x_std, training_dropped_y), verbosity = 4)

MLJ.save(joinpath(machines_folder,"mach_Neuralnetwork_tuned_server3.jlso"), mach_Neuralnetwork_tuned)

# update_loss(loss) = push!(losses, loss)
# update_training_loss(losses) = push!(training_losses, losses[end])
# update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
# update_epochs(epoch) = push!(epochs, epoch)

# save_control =
#     MLJIteration.skip(Save(joinpath(losses_folder, "neural_network_losses.jlso")), predicate=3)

# controls=[Step(2),
#           Patience(3),
#           InvalidValue(),
#           TimeLimit(5/60),
#           save_control,
#           WithLossDo(),
#           WithLossDo(update_loss),
#           WithTrainingLossesDo(update_training_loss),
#           Callback(update_means),
#           WithIterationsDo(update_epochs)
# ];

# plot(epochs, losses,
#      xlab = "epoch",
#      ylab = "root squared error",
#      label="out-of-sample")
# plot!(epochs, training_losses, label="training")

# savefig(joinpath(DIR, "loss.png"))
#predict_only_mach = machine("mach_Neuralnetwork_tuned_server1.jlso")
#rep = report(predict_only_mach).best_history_entry.model
#plot(predict_only_mach)
#scatter(reshape(rep.plotting.parameter_values, :), rep.plotting.measurements, xlabel = "K", ylabel = "AUC")

#print("best fitted parameters: " , fitted_params(mach_Neuralnetwork_tuned).best_model, "\n")

#fitted_params(mach_Neuralnetwork_tuned)
#(report(mach_Neuralnetwork_tuned))
print("tuning parameters: ", tuned_model_Neuralnetwork.range)
model_report = report(mach_Neuralnetwork_tuned).best_history_entry.model
print(model_report)

# param_df = Dataframe("parameters" = ) save parameters in a file ?
print("fitted parameters: \n", "epochs = ", model_report.epochs, "\n", "batch_size = ", model_report.batch_size, "\n", "lambda = ", model_report.lambda, "\n", "alpha = ", model_report.alpha, "\n")

pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, training_dropped_x_std)

err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_dropped_y)

print("Neural Network: ", err_rate_Neuralnewtwork, "\n")

proba_Neuralnetwork = predict(mach_Neuralnetwork_tuned, test_data_std)
prediction_Neuralnetwork_df = DataFrame(id = 1:nrow(test_data_std), precipitation_nextday = broadcast(pdf,proba_Neuralnetwork, true))
write_csv("neural_newtork_server3.csv", prediction_Neuralnetwork_df)


# dans le rapport: pourquoi avoir choisi quelle méthode, expliquer où on a passé du temps pourquoi et comment on a résolu 
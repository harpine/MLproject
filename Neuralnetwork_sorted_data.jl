#include("./first_code.jl")
include("utilities.jl")
include("./datasets.jl")
#include("loss_saver.jl")

machines_folder = "machines"
mkpath(machines_folder)

machine_subname = "sorted_mlp1"
# model_Neuralnetwork = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
#                                                                                 #Dense(100,100,relu),
#                                                                                 #Dense(100,100,relu),
#                                                                                 Dense(100, n_out, sigmoid))),
#                                                                                 optimiser = ADAMW()) # , finaliser = NNlib.sigmoid: I can't implement it!!

# christmas tree 
# 

model_Neuralnetwork = NeuralNetworkClassifier(
	builder = MLJFlux.MLP(hidden=(100,100,100)),
	optimiser = ADAMW(),
	lambda = 0.0,
	alpha = 0.0, batch_size = 32, finaliser = NNlib.softmax)

# model_Neuralnetwork = NeuralNetworkClassifier(
# 	builder = MLJFlux.Short(),
# 	optimiser = ADAMW(),
# 	lambda = 0.0,
# 	alpha = 0.0, finaliser = NNlib.softmax)


#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [23,24]),range(model_Neuralnetwork, :(builder.n_hidden), values = [24,25,26,27]), range(model_Neuralnetwork, :batch_size, values = [80,85,90])])#, acceleration=CUDALibs()) #, tune: optimiser, 


"""
tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5,7,10,15]), range(model_Neuralnetwork, :lambda, lower = 2e-6 , upper = 2e-2, scale = :log), range(model_Neuralnetwork, :alpha, values = [0,0.5,1.0] )])#, acceleration=CUDALibs()) #, tune: optimiser, 
"""
#server4
tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :batch_size, values = [32,64,128]), range(model_Neuralnetwork, :(epochs), values = [10,20,30,40,50,60,70,80]),range(model_Neuralnetwork, :(builder.hidden), values = [(80,50,30), (50,30,10), (60,45,30)])])#, acceleration=CUDALibs()) #, tune: optimiser, 
#builder mlp

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [15, 17, 20, 22, 25]),range(model_Neuralnetwork, :(builder.n_hidden), values = [10, 12, 15, 20, 30])])#, acceleration=CUDALibs()) #, tune: optimiser, 
#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [ range(model_Neuralnetwork, :lambda, lower = 2e-3 , upper = 2e3, scale = :log), range(model_Neuralnetwork, :alpha, lower = 0, upper = 1 )])#, acceleration=CUDALibs()) #, tune: optimiser, 

regularized_training_filled_x_std
mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, regularized_training_filled_x_std, training_filled_y), verbosity = 4)

MLJ.save(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"), mach_Neuralnetwork_tuned)

# predict_only_mach = machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_pc1.jlso"))
# rep = report(predict_only_mach).best_history_entry.model
plot(mach_Neuralnetwork_tuned)

print("tuning parameters: ", tuned_model_Neuralnetwork.range, "\n")
model_report = report(mach_Neuralnetwork_tuned).best_history_entry.model
print(model_report)

# param_df = Dataframe("parameters" = ) save parameters in a file ?
print("fitted parameters: \n", "epochs = ", model_report.epochs, "\n", "batch_size = ", model_report.batch_size, "\n", "lambda = ", model_report.lambda, "\n", "alpha = ", model_report.alpha, "\n", "n_hidden =", model_report.builder.n_hidden, "\n")

pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, regularized_training_filled_x_std)

err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

print("Neural Network: ", err_rate_Neuralnewtwork, "\n")

proba_Neuralnetwork = predict(mach_Neuralnetwork_tuned, regularized_test_x_std)
prediction_Neuralnetwork_df = DataFrame(id = 1:nrow(regularized_test_x_std), precipitation_nextday = broadcast(pdf,proba_Neuralnetwork, true))
write_csv("neural_newtork_tuned_" * machine_subname * ".csv", prediction_Neuralnetwork_df)


report(mach_Neuralnetwork_tuned).best_history_entry.measurement
report(mach_Neuralnetwork_tuned).best_history_entry

save_statistics_neuronal(machine_subname, tuned_model_Neuralnetwork, mach_Neuralnetwork_tuned, true, true)
plot(mach_Neuralnetwork_tuned)
savefig(joinpath(plots_folder, "machine_plot" * machine_subname * "png"))
#loss_saver(mach_Neuralnetwork_tuned)

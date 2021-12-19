#include("./first_code.jl")
include("./datasets.jl")
#include("loss_saver.jl")

machines_folder = "machines"
mkpath(machines_folder)

#machine_subname = "sorted_mlp_2layers_6"
machine_subname = "sorted_mlp_2layers_7_CV20"
# model_Neuralnetwork = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
#                                                                                 #Dense(100,100,relu),
#                                                                                 #Dense(100,100,relu),
#                                                                                 Dense(100, n_out, sigmoid))),
#                                                                                 optimiser = ADAMW()) # , finaliser = NNlib.sigmoid: I can't implement it!!

# christmas tree 
# 

builder_type = "mlp"

model_Neuralnetwork = NeuralNetworkClassifier(
	builder = MLJFlux.MLP(hidden=(5,3)),
	optimiser = ADAMW(),
	lambda = 0.0,
	alpha = 0.0, batch_size = 32, finaliser = NNlib.softmax, epochs = 5)



#builder_type = "short"

# model_Neuralnetwork = NeuralNetworkClassifier(
# 	builder = MLJFlux.Short(n_hidden = 25),
# 	optimiser = ADAMW(),
# 	lambda = 0.0,
# 	alpha = 0.0, finaliser = NNlib.softmax, batch_size = 85, epochs = 23)

# short builder : 
#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :epochs, values = [20,21,22,23,24,25,26,27,28]), range(model_Neuralnetwork, :batch_size, values = [80,85,90,95]), range(model_Neuralnetwork, :(builder.n_hidden), values = [23,24,25,26,27]), range(model_Neuralnetwork, :(builder.dropout), values = [0.0,0.2,0.5,0.7])])#, acceleration=CUDALibs()) #, tune: optimiser, 


"""
tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [5,7,10,15]), range(model_Neuralnetwork, :lambda, lower = 2e-6 , upper = 2e-2, scale = :log), range(model_Neuralnetwork, :alpha, values = [0,0.5,1.0] )])#, acceleration=CUDALibs()) #, tune: optimiser, 
"""
#server4
#mlp_2layers_6_CV20:
tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :batch_size, values = [32,48,64,86,100,128]),range(model_Neuralnetwork, :epochs, values = [20,24,27,28,29,30,35]) ,range(model_Neuralnetwork, :(builder.hidden), values = [(100,30), (100,40)])])#, acceleration=CUDALibs()) #, tune: optimiser, 

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :epochs, values = [26,35]) ,range(model_Neuralnetwork, :(builder.hidden), values = [(100,30), (100,40)])])#, acceleration=CUDALibs()) #, tune: optimiser, 


#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :epochs, values = [4,5,6,7]),range(model_Neuralnetwork, :(builder.hidden), values = [(30,20,10), (100,40,10), (100,25,6)])])#, acceleration=CUDALibs()) #, tune: optimiser, 
#builder mlp
#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 20), measure = auc, range = [range(model_Neuralnetwork, :batch_size, values = [20,32,50,70,90,110]) ,range(model_Neuralnetwork, :(epochs), lower = 5, upper = 80),range(model_Neuralnetwork, :(builder.hidden), values = [(20,10), (15,7), (25,15), (10,5)])])#, acceleration=CUDALibs()) #, tune: optimiser, 

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :batch_size, values = [50,55,60]) ,range(model_Neuralnetwork, :(epochs), values = [65,70,75])]) #,range(model_Neuralnetwork, :(builder.hidden), values = [(100,80,60), (60,45,30)])])#, acceleration=CUDALibs()) #, tune: optimiser, 

#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [range(model_Neuralnetwork, :(epochs), values = [15, 17, 20, 22, 25]),range(model_Neuralnetwork, :(builder.n_hidden), values = [10, 12, 15, 20, 30])])#, acceleration=CUDALibs()) #, tune: optimiser, 
#tuned_model_Neuralnetwork = TunedModel(model = model_Neuralnetwork, resampling= CV(nfolds = 10), measure = auc, range = [ range(model_Neuralnetwork, :lambda, lower = 2e-3 , upper = 2e3, scale = :log), range(model_Neuralnetwork, :alpha, lower = 0, upper = 1 )])#, acceleration=CUDALibs()) #, tune: optimiser, 

mach_Neuralnetwork_tuned = fit!(machine(tuned_model_Neuralnetwork, regularized_training_filled_x_std, training_filled_y), verbosity = 4)

MLJ.save(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"), mach_Neuralnetwork_tuned)

# predict_only_mach = machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_pc1.jlso"))
# rep = report(predict_only_mach).best_history_entry.model
plot(mach_Neuralnetwork_tuned)

print("tuning parameters: ", tuned_model_Neuralnetwork.range, "\n")
model_report = report(mach_Neuralnetwork_tuned).best_history_entry.model
print(model_report)

# param_df = Dataframe("parameters" = ) save parameters in a file ?

print("fitted parameters: \n", "epochs = ", model_report.epochs, "\n", "batch_size = ", model_report.batch_size, "\n", "lambda = ", model_report.lambda, "\n", "alpha = ", model_report.alpha, "\n")
if builder_type == "short"
	print("n_hidden = ", model_report.builder.n_hidden, "\n")
elseif builder_type == "mlp"
	print("hidden = ", model_report.builder.hidden, "\n")
end


pred_Neuralnetwork = predict_mode(mach_Neuralnetwork_tuned, regularized_training_filled_x_std)

err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

print("Neural Network: ", err_rate_Neuralnewtwork, "\n")

proba_Neuralnetwork = predict(mach_Neuralnetwork_tuned, regularized_test_std)
prediction_Neuralnetwork_df = DataFrame(id = 1:nrow(regularized_test_std), precipitation_nextday = broadcast(pdf,proba_Neuralnetwork, true))
write_csv("neural_newtork_tuned_" * machine_subname * ".csv", prediction_Neuralnetwork_df)

print("AUC on training:", area_under_curve(predict(mach_Neuralnetwork_tuned, regularized_training_filled_x_std), training_filled_y))


report(mach_Neuralnetwork_tuned).best_history_entry.measurement
report(mach_Neuralnetwork_tuned).best_history_entry.model

save_statistics_neuronal(machine_subname, tuned_model_Neuralnetwork, mach_Neuralnetwork_tuned, short_builder = false, mlp_builder = true, regularized = true)
plot(mach_Neuralnetwork_tuned)
savefig(joinpath(plots_folder, "machine_plot" * machine_subname * ".png"))
#loss_saver(mach_Neuralnetwork_tuned)

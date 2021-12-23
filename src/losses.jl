include("datasets.jl")
machine_subname = "sorted_mlp_2layers_6_CV20"
mach_Neuralnetwork_tuned = machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

#First way to plot losses : 

r = range(mach_Neuralnetwork_tuned.model.model, :epochs, lower = 1, upper = 80)

curve = learning_curve(mach_Neuralnetwork_tuned.model.model, training_filled_x_std, training_filled_y,
                       range=r,
                       resampling= Holdout(fraction_train=0.7), #CV(nfolds = 5),
                       measure=auc)


plot(curve.parameter_values,
       curve.measurements,
       xlab=curve.parameter_name,
       xscale=curve.parameter_scale,
       ylab = "AUC")

savefig(joinpath(losses_folder, "validation_loss_plot_" * machine_subname * ".png"))
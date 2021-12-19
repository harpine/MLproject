include("datasets.jl")
machine_subname = "sorted_mlp_2layers_6"
mach_Neuralnetwork_tuned = machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

area_under_curve(predict(mach_Neuralnetwork_tuned, regularized_training_filled_x_std), training_filled_y)


#First way to plot losses : 

r = range(mach_Neuralnetwork_tuned.model.model, :epochs, lower = 1, upper = 80)

curve = learning_curve(mach_Neuralnetwork_tuned.model.model, training_filled_x_std, training_filled_y,
                       range=r,
                       resampling= Holdout(fraction_train=0.7), #CV(nfolds = 5),
                       measure=auc)


using Plots
plot(curve.parameter_values,
       curve.measurements,
       xlab=curve.parameter_name,
       xscale=curve.parameter_scale,
       ylab = "AUC")

savefig(joinpath(losses_folder, "validation_loss_plot_" * machine_subname * ".png"))
# savefig(joinpath(losses_folder, "loss_plot_sorted.png"))



# # second way to plot losses:
# losses_folder = "losses"
# mkpath(losses_folder)


# parameters(mach_Neuralnetwork_tuned) = make1d.(Flux.params(fitted_params(mach_Neuralnetwork_tuned)));

# controls_folder = "controls"
# mkpath(joinpath(losses_folder,controls_folder))
# # To update the traces:

# update_loss(loss) = push!(losses, loss)
# update_training_loss(losses) = push!(training_losses, losses[end])
# update_means(mach_Neuralnetwork_tuned) = append!(parameter_means, mean.(parameters(mach_Neuralnetwork_tuned)));
# update_epochs(epoch) = push!(epochs, epoch)

# # The controls to apply:

# save_control = MLJIteration.skip(Save(joinpath(losses_folder, "controls", "controls_" * machine_subname * ".jlso")), predicate=3)

# controls=[Step(2),
#         Patience(10),
#         InvalidValue(),
#         TimeLimit(10/60),
#         save_control,
#         WithLossDo(),
#         WithLossDo(update_loss),
#         WithTrainingLossesDo(update_training_loss),
#         Callback(update_means),
#         WithIterationsDo(update_epochs)
# ];

# losses = []
# training_losses = []
# parameter_means = Float32[];
# epochs = []

# iterated_clf = IteratedModel(model=mach_Neuralnetwork_tuned.model.model,
#                     controls=controls,
#                     resampling= Holdout(fraction_train=0.7),
#                     measure=log_loss)


# make2d(x::AbstractArray) = reshape(x, :, size(x)[end])
# make1d(x::AbstractArray) = reshape(x, length(x));
# # ### Binding the wrapped model to data: + training:

# mach = fit!(machine(iterated_clf, training_filled_x_std, training_filled_y), verbosity = 4)


# # ### Comparison of the training and out-of-sample losses:
# plot(epochs, losses,
#     xlab = "epoch",
#     ylab = "log_loss",
#     label="out-of-sample")
# plot!(epochs, training_losses, label="training")

# print(training_losses)
# savefig(joinpath(losses_folder, "loss_plot_" * machine_subname * ".png"))


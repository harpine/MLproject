include("./datasets.jl")

machine_subname = "sorted_mlp_2layers_6_CV5"
mach_cv5layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

machine_subname = "sorted_mlp_2layers_6_CV20"
mach_cv20layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

size(regularized_training_filled_x_std)[1] / 5

train1 = regularized_training_filled_x_std[1:635,:]
train2 = regularized_training_filled_x_std[635:1270,:]
train3 = regularized_training_filled_x_std[1270:1905,:]
train4 = regularized_training_filled_x_std[1905:2540,:]
train5 = regularized_training_filled_x_std[2540:end,:]

train1y = training_filled_y[1:635]
train2y = training_filled_y[635:1270]
train3y = training_filled_y[1270:1905]
train4y = training_filled_y[1905:2540]
train5y = training_filled_y[2540:end]

size(train1)
size(train1y)
print(typeof(train1y))
auc_cv5 = [area_under_curve(predict(mach_cv5layers, train1), train1y), area_under_curve(predict(mach_cv5layers, train2), train2y), area_under_curve(predict(mach_cv5layers, train3), train3y), area_under_curve(predict(mach_cv5layers, train4), train4y), area_under_curve(predict(mach_cv5layers, train5), train5y)]
auc_cv20 = [area_under_curve(predict(mach_cv20layers, train1), train1y), area_under_curve(predict(mach_cv20layers, train2), train2y), area_under_curve(predict(mach_cv20layers, train3), train3y), area_under_curve(predict(mach_cv20layers, train4), train4y), area_under_curve(predict(mach_cv20layers, train5), train5y)]

datafr = vcat(DataFrame(auc = auc_cv5, type = "CV5"), DataFrame(auc = auc_cv20, type = "cv20"))
PlotlyJS.plot(datafr, x=:type, y=:auc, kind="box", boxmean = true, boxpoints = "all")


include("datasets.jl")


import PlotlyJS

machine_subname = "sorted_mlp_2layers_6"
mach_2layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

plot(mach_2layers)

per_fold_2layers = report(mach_2layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV5"
mach_cv5layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv5layers = report(mach_cv5layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV10_control"
mach_cv10layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv10layers = report(mach_cv10layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV20"
mach_cv20layers= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv20layers = report(mach_cv20layers).best_history_entry.per_fold[1]

# machine_subname_8 = "sorted_mlp_2layers_8_CV20"
# mach_2layers_8= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname_8 * ".jlso"))

# per_fold_2layers_8 = report(mach_2layers_8).best_history_entry.per_fold[1]

# machine_subname_7 = "sorted_mlp_2layers_7_CV20"
# mach_2layers_7= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * machine_subname_7 * ".jlso"))

# per_fold_2layers_7 = report(mach_2layers_7).best_history_entry.per_fold[1]

mach_subname = "sorted_short10"
mach_short= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_short = report(mach_short).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV5"
mach_shortcv5= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv5 = report(mach_shortcv5).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV20_seed5"
mach_shortcv20_seed5= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv20_seed5 = report(mach_shortcv20_seed5).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV10"
mach_shortcv10= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv10 = report(mach_shortcv10).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV20"
mach_shortcv20= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv20 = report(mach_shortcv20).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV5_seed5"
mach_shortcv5_seed5= machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv5_seed5 = report(mach_shortcv5_seed5).best_history_entry.per_fold[1]

mach_subname = "l2"
mach_l2= machine(joinpath(machines_folder,"mach_logistic_reg_" * mach_subname * ".jlso"))

per_fold_l2 = report(mach_l2).best_history_entry.per_fold[1]






datafr = vcat(DataFrame(auc = per_fold_l2, type = "L2"), DataFrame(auc = per_fold_cv5layers, type = "MLP6_CV5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_cv10layers, type = "MLP6_CV10"))
datafr = vcat(datafr, DataFrame(auc = per_fold_cv20layers, type = "MLP6_CV20"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv5, type = "SHORT_CV5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv5_seed5, type = "SHORT_CV5_seed5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv10, type = "SHORT_CV10"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv20, type = "SHORT_CV20"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv20_seed5, type = "SHORT_CV20_seed5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_2layers, type = "MLP6"))


print("2 layers : " , std(per_fold_2layers), " short : ", std(per_fold_short))

p = PlotlyJS.plot(datafr, x=:type, y=:auc, kind="box", boxmean = true, boxpoints = "all")

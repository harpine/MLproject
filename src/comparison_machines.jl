include("datasets.jl")

import PlotlyJS
import PlotlyBase: savefig

per_fold_2layers = report(mach_2layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV5"
mach_cv5layers= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv5layers = report(mach_cv5layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV10_control"
mach_cv10layers= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv10layers = report(mach_cv10layers).best_history_entry.per_fold[1]

machine_subname = "sorted_mlp_2layers_6_CV20"
mach_cv20layers= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * machine_subname * ".jlso"))

per_fold_cv20layers = report(mach_cv20layers).best_history_entry.per_fold[1]


mach_subname = "sorted_short10_1_CV5"
mach_shortcv5= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv5 = report(mach_shortcv5).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV10"
mach_shortcv10= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv10 = report(mach_shortcv10).best_history_entry.per_fold[1]

mach_subname = "sorted_short10_1_CV20"
mach_shortcv20= machine(joinpath(machines_folder_plot,"mach_Neuralnetwork_tuned_" * mach_subname * ".jlso"))

per_fold_shortcv20 = report(mach_shortcv20).best_history_entry.per_fold[1]

mach_subname = "l2"
mach_l2= machine(joinpath(machines_folder_plot,"mach_logistic_reg_" * mach_subname * ".jlso"))

per_fold_l2 = report(mach_l2).best_history_entry.per_fold[1]

datafr = vcat(DataFrame(auc = per_fold_l2, type = "L2"), DataFrame(auc = per_fold_cv5layers, type = "MLP6_CV5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_cv10layers, type = "MLP6_CV10"))
datafr = vcat(datafr, DataFrame(auc = per_fold_cv20layers, type = "MLP6_CV20"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv5, type = "SHORT_CV5"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv10, type = "SHORT_CV10"))
datafr = vcat(datafr, DataFrame(auc = per_fold_shortcv20, type = "SHORT_CV20"))


print(  "mlp6_cv5 : " , std(per_fold_cv5layers), "\n",
        "mlp6_cv10 : " , std(per_fold_cv10layers), "\n",
        "mlp6_cv20 : " , std(per_fold_cv20layers), "\n",
        "short_cv5 : " , std(per_fold_shortcv5), "\n",
        "short_cv10 : " , std(per_fold_shortcv10), "\n",
        "short_cv20 : " , std(per_fold_shortcv20), "\n")

p = PlotlyJS.plot(datafr, x=:type, y=:auc, yaxis=(title="AUC"), kind="box", boxmean = true, boxpoints = "all")

PlotlyBase.savefig(p, joinpath(plots_folder,"boxplot.png"))
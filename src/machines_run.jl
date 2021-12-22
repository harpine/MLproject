
machines_dictionnary = Dict("logistic_reg" => machine_subname -> logistic_reg_l2(machine_subname),
                            "knn" => machine_subname -> knn(machine_subname), 
                            "random_forest" => machine_subname -> random_forest(machine_subname), 
                            "short_neuralnetwork" => machine_subname -> short_neuralnetwork(machine_subname), 
                            "mlp_neuralnetwork" => machine_subname -> mlp_neuralnetwork(machine_subname))


"""
*(type_machine, machine_subname, best_machines = true) # = true à mettre dans la docstring????
Allow to choose a machine, run it, and save the different outptuts, with the subname of the machine.
The possibilities of machine type are: "Logistic_l1" , "Logistic_l2", "KNN" "Short_Neuralnetwork", "Mlp_Neuralnetwork", "RandomForest"
"""
function run_machine(type_machine, machine_subname)
    machines_dictionnary[type_machine](machine_subname)
end

best_models = ["short_neuralnetwork", "mlp_neuralnetwork"]

"""
*(machines, single_subname)
allow to tune and apply some machines; 
"all" = all machines
"best" = best machines 
"logistic_reg", "knn", "random_forest", "short_neuralnetwork" or "mlp_neuralnetwork" = run only the given machine. 
If you chose to run one specific type of machine, you can specify the string subname of the machine in the second argument. 
"""
function run_machines(machines, single_subname  = "tuned") 
    if machines == "all"
        for (type, func) in x
            run_machine(type, single_subname)
        end
    elseif machines == "best"
        for type in best_models
            run_machine(type, "best")
        end
    else
        run_machine(machines, single_subname)
    end

end

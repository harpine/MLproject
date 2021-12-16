include("./first_code.jl")

function save_statistics_neuronal(machine_subname, tuned_model, machine)
    
    tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    ep = model_rep.epochs
    batch = model_rep.batch_size
    lamb = model_rep.lambda
    alph = model_rep.alpha
    n_hidd = model_rep.builder.n_hidden

    measure = report(machine).best_history_entry.measurement
    pred_Neuralnetwork = predict_mode(machine, regularized_training_filled_x_std)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = tuning_param, model_report = model_rep, epochs = ep, batch_size = batch, lambda = lamb, alpha = alph, n_hidden = n_hidd, validation_measure_auc = measure, error_rate = err_rate_Neuralnewtwork)
    write_stat(machine_subname * ".csv", stats)
end


function save_statistics_neuronal_old(machine_subname, machine, regularized = false)
    
    #tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    ep = model_rep.epochs
    batch = model_rep.batch_size
    lamb = model_rep.lambda
    alph = model_rep.alpha
    #n_hidd = model_rep.builder.n_hidden
    data = training_filled_x_std
    if regularized
        data = regularized_training_filled_x_std
    end
    pred_Neuralnetwork = predict_mode(machine, data)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

    measure = report(machine).best_history_entry.measurement

    stats = DataFrame(machine = machine_subname, tuning_parameters = " ", model_report = model_rep, epochs = ep, batch_size = batch, lambda = lamb, alpha = alph, validation_measure_auc = measure, error_rate = err_rate_Neuralnewtwork)
    write_stat(machine_subname * ".csv", stats)
end

test_mach = machine(joinpath(machines_folder,"mach_Neuralnetwork_tuned_pc2.jlso"))
save_statistics_neuronal_old("pc2", test_mach)

function save_statistics_randomForest(machine_subname, tuned_model, machine)
    
    tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    tree = model_rep.n_trees
    depth = model_rep.max_depth
    min = model_rep.min_samples_split

    measure = report(machine).best_history_entry.measurement
    pred_randomForest = predict_mode(machine, training_filled_x_std)
    err_rate_randomForest = mean(pred_randomForest .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = tuning_param, model_report = model_rep, n_trees = tree, max_depth = depth, min_samples_split = min, validation_measure_auc = measure, error_rate = err_rate_randomForest)
    write_stat(machine_subname * ".csv", stats)
end

function save_statistics_randomForest_old(machine_subname, machine)

    model_rep = report(machine).best_history_entry.model
    tree = model_rep.n_trees
    depth = model_rep.max_depth
    min = model_rep.min_samples_split

    measure = report(machine).best_history_entry.measurement
    pred_randomForest = predict_mode(machine, training_filled_x)
    err_rate_randomForest = mean(pred_randomForest .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = " ", model_report = model_rep, n_trees = tree, max_depth = depth, min_samples_split = min, validation_measure_auc = measure, error_rate = err_rate_randomForest)
    write_stat(machine_subname * ".csv", stats)
end

test_mach = machine(joinpath(machines_folder,"mach_RandomForest_filled_server2.jlso"))
save_statistics_randomForest_old("RandomForest_server2", test_mach)
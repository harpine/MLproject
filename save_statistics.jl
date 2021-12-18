#include("./utilities.jl")

function save_statistics_neuronal(machine_subname, tuned_model, machine, ; short_builder = false, mlp_builder = false, regularized = false)
    
    tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    ep = model_rep.epochs
    batch = model_rep.batch_size
    lamb = model_rep.lambda
    alph = model_rep.alpha
    n_hidd = "not appliable"
    if short_builder
        n_hidd = model_rep.builder.n_hidden
    end
    hidd = "not appliable"
    if mlp_builder
        hidd = model_rep.builder.hidden
    end
    data = training_filled_x_std
    if regularized
        data = regularized_training_filled_x_std
    end
    measure = report(machine).best_history_entry.measurement
    pred_Neuralnetwork = predict_mode(machine,data)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = tuning_param, model_report = model_rep, epochs = ep, batch_size = batch, lambda = lamb, alpha = alph, n_hidden = n_hidd, hiddden = hidd, validation_measure_auc = measure, error_rate = err_rate_Neuralnewtwork)
    write_stat("Neuralnetwork_" * machine_subname * ".csv", stats)
end


function save_statistics_neuronal_old(machine_subname, machine, short_builder = false, regularized = false)

    model_rep = report(machine).best_history_entry.model
    ep = model_rep.epochs
    batch = model_rep.batch_size
    lamb = model_rep.lambda
    alph = model_rep.alpha
    n_hidd = "not appliable"
    if short_builder
        n_hidd = model_rep.builder.n_hidden
    end
    data = training_filled_x_std
    if regularized
        data = regularized_training_filled_x_std
    end
    pred_Neuralnetwork = predict_mode(machine, data)
    err_rate_Neuralnewtwork = mean(pred_Neuralnetwork .!= training_filled_y)

    measure = report(machine).best_history_entry.measurement

    stats = DataFrame(machine = machine_subname, tuning_parameters = " ", model_report = model_rep, epochs = ep, batch_size = batch, lambda = lamb, alpha = alph, validation_measure_auc = measure, error_rate = err_rate_Neuralnewtwork)
    write_stat("Neuralnetwork_" * machine_subname * ".csv", stats)
end

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
    write_stat("RandomForest_" * machine_subname * ".csv", stats)
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
    write_stat("RandomForest_" * machine_subname * ".csv", stats)
end

function save_statistics_KNN_class(machine_subname, tuned_model, machine, standardized = false, regularized = false)
    
    tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    K_value = model_rep.K

    measure = report(machine).best_history_entry.measurement

    if standardized & regularized
        pred_KNN_class = predict_mode(machine, regularized_training_filled_x_std)
    elseif regularized & !standardized
        pred_KNN_class = predict_mode(machine, regularized_training_filled_x)
    elseif standardized & !regularized
        pred_KNN_class = predict_mode(machine, training_filled_x_std)
    else
        pred_KNN_class = predict_mode(machine, training_filled_x)
    end
    err_rate_KNN_class = mean(pred_KNN_class .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = tuning_param, model_report = model_rep, K = K_value, validation_measure_auc = measure, error_rate = err_rate_KNN_class)
    write_stat("KNN_class_ " * machine_subname * ".csv", stats)
end


function save_logistic_reg(machine_subname, tuned_model, machine)
    
    tuning_param = [tuned_model.range]

    model_rep = report(machine).best_history_entry.model
    lamb = model_rep.lambda

    measure = report(machine).best_history_entry.measurement

    pred_logistic_class = predict_mode(machine, training_filled_x_std)

    err_rate_logistic_class = mean(pred_logistic_class .!= training_filled_y)

    stats = DataFrame(machine = machine_subname, tuning_parameters = tuning_param, model_report = model_rep, lambda = lamb, validation_measure_auc = measure, error_rate = err_rate_logistic_class)
    write_stat("logistic_reg_ " * machine_subname * ".csv", stats)
end


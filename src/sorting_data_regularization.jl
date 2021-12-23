import GLMNet: glmnet

function sort_data_std(training_filled_x_std, training_filled_y, test_data_std)

    training_fits = glmnet(Array(training_filled_x_std), training_filled_y)

    lambda = log.(training_fits.lambda)
    small = []
    col_names = names(training_filled_x_std)

    idx = findall(x->x<=-8.0, lambda)[1]
    for i in 1:size(training_fits.betas, 1)
        if abs(training_fits.betas[i, idx]) < 1e-8
            push!(small, col_names[i])
        end
    end
    
    regularized_training_filled_x_std = select(training_filled_x_std, Not(small))
    regularized_test_x_std = select(test_data_std, Not(small))

    write_preprocess_data(regularized_training_filled_x_std_name, regularized_training_filled_x_std)
    write_preprocess_data(regularized_test_std_name, regularized_test_x_std)
end
#Not standardized 

function sort_data_non_std(training_filled_x, training_filled_y, test_data)
    
    training_fits = glmnet(Array(training_filled_x), training_filled_y)

    lambda = log.(training_fits.lambda)
    small = []
    col_names = names(training_filled_x)
    idx = findall(x->x<=-8.0, lambda)[1]
    for i in 1:size(training_fits.betas, 1)
        plot!(lambda, training_fits.betas[i, :], label = col_names[i])
        if abs(training_fits.betas[i, idx]) < 1e-8
            push!(small, col_names[i])
        end
    end

    regularized_training_filled_x = select(training_filled_x, Not(small))
    regularized_test_x = select(test_data, Not(small))

    write_preprocess_data(regularized_training_filled_x_name, regularized_training_filled_x)
    write_preprocess_data(regularized_test_name, regularized_test_x)
end



function write_normalized_regularized_data(training_data_x, test_data)
    data = training_data_x
    data_for_test = test_data
    for i in 1:(size(data,2))
        # Conversion of all columns into Float64
        if typeof(data[:, i][1]) != Float64
            data[!, i] = Float64.(data[:, i])
            data_for_test[!, i] = Float64.(data_for_test[:,i])
        end
        mini = minimum(data[:, i])
        maxi = maximum(data[:, i])
        data[:, i] = (data[:, i] .- mini) ./ (maxi - mini)
        data_for_test[:, i] = (data_for_test[:, i] .- mini) ./ (maxi - mini)
    end
    
    write_preprocess_data(regularized_training_filled_x_norm_name, data)
    write_preprocess_data(regularized_test_norm_name, data_for_test)
end


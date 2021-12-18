include("./utilities.jl")
# include("./first_code.jl")

# plotting lasso path:

import GLMNet: glmnet

function sort_data_std(training_filled_x_std, training_filled_y)
    
    # weather_input = select(weather, Not(:LUZ_wind_peak))[1:end-5, :]
    # weather_output = weather.LUZ_wind_peak[6:end]
    training_fits = glmnet(Array(training_filled_x_std), training_filled_y)

    lambda = log.(training_fits.lambda)
    small = []
    col_names = names(training_filled_x_std)
    plotly()
    p = plot()
    idx = findall(x->x<=0.77, lambda)[1]
    for i in 1:size(training_fits.betas, 1)
        plot!(lambda, training_fits.betas[i, :], label = col_names[i])
        if abs(training_fits.betas[i, idx]) < 1e-8
            push!(small, col_names[i])
        end
    end
    plot!(legend = :outertopright, xlabel = "log(λ)", size = (700, 400))
    gr()
    p

    regularized_training_filled_x_std = select(training_filled_x_std, Not(small))
    regularized_test_x_std = select(test_data_std, Not(small))


    write_preprocess_data(regularized_training_filled_x_std_name, regularized_training_filled_x_std)
    write_preprocess_data(regularized_test_std_name, regularized_test_x_std)
end
#Not standardized 

function sort_data_non_std(training_filled_x, training_filled_y)
    
    training_fits = glmnet(Array(training_filled_x), training_filled_y)

    lambda = log.(training_fits.lambda)
    small = []
    col_names = names(training_filled_x)
    plotly()
    p = plot()
    idx = findall(x->x<=0.77, lambda)[1]
    for i in 1:size(training_fits.betas, 1)
        plot!(lambda, training_fits.betas[i, :], label = col_names[i])
        if abs(training_fits.betas[i, idx]) < 1e-8
            push!(small, col_names[i])
        end
    end
    plot!(legend = :outertopright, xlabel = "log(λ)", size = (700, 400))
    gr()
    p

    regularized_training_filled_x = select(training_filled_x, Not(small))
    regularized_test_x = select(test_data, Not(small))

    write_preprocess_data(regularized_training_filled_x_name, regularized_training_filled_x)
    write_preprocess_data(regularized_test_name, regularized_test_x)
end


#sort_data_std(training_filled_x_std, training_filled_y)
# sort_data_non_std(training_filled_x, training_filled_y)



function normalize_regularized_data(data)
    for i in 1:(size(data,2))
        if typeof(data[:, i][1]) != Float64
            data[!, i] = Float64.(data[:, i])
        end
        mini = minimum(data[:, i])
        maxi = maximum(data[:, i])
        data[:, i] = (data[:, i] .- mini) ./ (maxi - mini)
    end

end

function write_normalized_regularized_data(training_data_x, test_data)
    write_preprocess_data(regularized_training_filled_x_norm_name, normalize_regularized_data(training_data_x))
    write_preprocess_data(regularized_test_norm_name, normalize_regularized_data(test_data))
end


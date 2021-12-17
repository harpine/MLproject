include("./utilities.jl")

# plotting lasso path:

import GLMNet: glmnet

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
    if abs(training_fits.betas[i, 73]) < 1e-8
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

#Not standardized 
training_fits = glmnet(Array(training_filled_x), training_filled_y)

lambda = log.(training_fits.lambda)
small = []
col_names = names(training_filled_x)
plotly()
p = plot()
idx = findall(x->x<=0.77, lambda)[1]
for i in 1:size(training_fits.betas, 1)
    plot!(lambda, training_fits.betas[i, :], label = col_names[i])
    if abs(training_fits.betas[i, 73]) < 1e-8
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

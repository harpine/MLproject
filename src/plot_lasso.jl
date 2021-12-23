include("./datasets.jl")

import GLMNet: glmnet

training_fits = glmnet(Array(training_filled_x), training_filled_y)

lambda = log.(training_fits.lambda)
small = []
col_names = names(training_filled_x)
plotly()
p = plot()
idx = findall(x->x<=-8.0, lambda)[1]
importants = []
for i in 1:size(training_fits.betas, 1)
    if abs(training_fits.betas[i, idx]) > 0.25
        push!(importants, (abs(training_fits.betas[i, idx]),i))
    else
        plot!(lambda, training_fits.betas[i, :], label = "")
    end
    if abs(training_fits.betas[i, idx]) < 1e-8
        push!(small, col_names[i])
    end

end

importants = sort(importants, rev = true)
for (idx, val) in enumerate(importants)
    plot!(lambda, training_fits.betas[val[2], :], label = col_names[val[2]])
end

plot!(legend = :inside, xlabel = "log(λ)", size = (1000, 700))
gr()
p
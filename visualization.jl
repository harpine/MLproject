
include("./datasets.jl")

# PCA VISUALIZATION
function biplot(m; pc = 1:2)
    scores = MLJ.transform(m, m.data[1])
    p = scatter(getproperty(scores, Symbol(:x, pc[1])),
                getproperty(scores, Symbol(:x, pc[2])),
                label = nothing, aspect_ratio = 1, size = (600, 600),
                xlabel = "PC$(pc[1])", ylabel = "PC$(pc[2])",
                framestyle = :axis, markeralpha = 0,
                txt = text.(1:nrows(scores), 8, :gray), markersize = 0)
    plot!(p[1], inset = (1, bbox(0, 0, 1, 1)),
          right_margin = 10Plots.mm, top_margin = 10Plots.mm)
    p2 = p[1].plt.subplots[end]
    plot!(p2, aspect_ratio = 1, mirror = true, legend = false)
    params = fitted_params(m)
    loadings = if hasproperty(params, :pca)
        params.pca.projection
    else
        params.projection
    end
    n = names(m.data[1])
    for i in 1:length(n)
        plot!(p2, [0, .9*loadings[i, pc[1]]], [0, .9*loadings[i, pc[2]]],
              c = :red, arrow = true)
        annotate!(p2, [(loadings[i, pc[1]], loadings[i, pc[2]],
                        (n[i], :red, :center, 8))])
    end
    scatter!(p2, 1.1*loadings[:, pc[1]], 1.1*loadings[:, pc[2]],
             markersize = 0, markeralpha = 0) # dummy for sensible xlims and xlims
    plot!(p2,
          background_color_inside = RGBA{Float64}(0, 0, 0, 0),
          tickfontcolor = RGB(1, 0, 0))
    p
end

pca_visualization = fit!(machine(PCA(), training_filled_x_std))
gr()
biplot(pca_visualization)
savefig(joinpath(plots_folder, "PCA_biplot.png"))


# CORRELATION PLOT
import GLMNet: glmnet

training_fits = glmnet(Array(training_filled_x_std), training_filled_y)

lambda = log.(training_fits.lambda)
small = []
col_names = names(training_filled_x_std)
idx = findall(x->x<=-8.0, lambda)[1]
importants = []
for i in 1:size(training_fits.betas, 1)
    if abs(training_fits.betas[i, idx]) > 0.25
        push!(importants, (abs(training_fits.betas[i, idx]),i))
    end
    if abs(training_fits.betas[i, idx]) < 1e-8
        push!(small, col_names[i])
    end
end
importants = sort(importants, rev = true)[1:5]
for (idx, val) in enumerate(importants)
    print(col_names[val[2]], ", ")
end
# The 5 most relevant predictors are PUY_air_temp_4, BER_air_temp_4, SIO_air_temp_4, NEU_air_temp_4, CDF_air_temp_4

# Correlation plot on the most relevant predictors, as chosen by the L1 regularization
@df training_filled_x_std corrplot([:PUY_air_temp_4 :BER_air_temp_4 :SIO_air_temp_4 :NEU_air_temp_4 :CDF_air_temp_4],
                     grid = false, fillcolor = cgrad(), size = (700, 700)) 
savefig(joinpath(plots_folder, "Corrplot.png"))

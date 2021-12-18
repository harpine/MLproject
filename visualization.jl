
include("./datasets.jl")

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




# Correlation plot
@df training_filled_x corrplot([:CHU_air_temp_1 :CHU_wind_1 :CHU_wind_direction_1 :DAV_radiation_1 :DAV_delta_pressure_1],
                     grid = false, fillcolor = cgrad(), size = (700, 700)) 
# Ca marche, maintenant trouver les directions les plus intéressantes avec PCA?

# DataFrame of the training data
schema(training_filled)


pca_visualization = fit!(machine(PCA(), training_filled_x_std))
gr()
biplot(pca_visualization, pc=(1,2))
savefig(joinpath(plots_folder, "PCA_biplot.png"))

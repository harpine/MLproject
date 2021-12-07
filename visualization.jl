
include("./first_code.jl")

# Correlation plot
@df training_dropped corrplot([:CHU_air_temp_1 :CHU_wind_1 :CHU_wind_direction_1 :DAV_radiation_1 :DAV_delta_pressure_1],
                     grid = false, fillcolor = cgrad(), size = (700, 700)) 
# Ca marche, maintenant trouver les directions les plus intéressantes avec PCA?

print(minimum(training_dropped.CHU_air_temp_1))
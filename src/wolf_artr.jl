#= ARTR neural data from Wolf et al 2023 =#
function artr_wolf_2023_folder(; temperature::Int)
    return @artifact_str("Wolf_ARTR_2023_T$temperature")
end

artr_wolf_2023_temperatures() = (18, 22, 26, 30, 33)

function artr_wolf_2023_mat(; temperature::Int, fish::Int)
    path = joinpath(artr_wolf_2023_folder(; temperature), "T$(temperature)_Fish $fish.mat")
    if isfile(path)
        return path
    else
        throw(ArgumentError("No such file: $path"))
    end
end

function artr_wolf_2023_fishes(; temperature::Int)
    dir = artr_wolf_2023_folder(; temperature)
    return map(readdir(dir)) do file
        parse(Int, first(split(last(split(file)), '.')))
    end
end

function artr_wolf_2023(; temperature::Int, fish::Int)
    return matread(artr_wolf_2023_mat(; temperature, fish))["Dinference_corr"]
end

# mat = matread(legoc2021_single_fish_T26_path())
# return [[filter(!isnan, traj) for traj = eachcol(trajs)] for trajs = vec(mat["dtheta"])]

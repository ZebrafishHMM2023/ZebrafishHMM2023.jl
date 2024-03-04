#= ARTR neural data from Wolf et al 2023 =#
function artr_wolf_2023_folder(temperature::Int)
    return @artifact_str("Wolf_ARTR_2023_T$temperature")
end

artr_wolf_2023_temperatures() = (18, 22, 26, 30, 33)

function artr_wolf_2023_temperatures(fish::Int)
    return [temperature for temperature = artr_wolf_2023_temperatures() if fish ∈ artr_wolf_2023_fishes(temperature)]
end

function artr_wolf_2023_mat(; temperature::Int, fish::Int)
    path = joinpath(artr_wolf_2023_folder(temperature), "T$(temperature)_Fish $fish.mat")
    if isfile(path)
        return path
    else
        throw(ArgumentError("No such file: $path"))
    end
end

function artr_wolf_2023_fishes(temperature::Int)
    dir = artr_wolf_2023_folder(temperature)
    return map(readdir(dir)) do file
        parse(Int, first(split(last(split(file)), '.')))
    end
end

function artr_wolf_2023_fishes()
    all_fish = [fish for temperature = artr_wolf_2023_temperatures() for fish = artr_wolf_2023_fishes(temperature)]
    return sort(unique(all_fish))
end

function artr_wolf_2023(; temperature::Int, fish::Int)
    return matread(artr_wolf_2023_mat(; temperature, fish))["Dinference_corr"]
end

function load_artr_wolf_2023(; temperature, fish)
    dict = artr_wolf_2023(; temperature, fish)
    return (;
        right = dict["rightspikesbin_data"]',
        left = dict["leftspikesbin_data"]',
        time = vec(dict["time"]),
        temperature = dict["T"]
    )
end

#= Distances between the neurons. Data kindly provided by S. Wolf (personal communication) =#
function artr_wolf_2023_distances_folder()
    return @artifact_str("Wolf_ARTR_2023_Distances")
end

function artr_wolf_2023_distances_file(; temperature::Int, fish::Int)
    return joinpath(artr_wolf_2023_distances_folder(), "dist_T$(temperature)_Fish $fish.mat")
end

function artr_wolf_2023_distances(; temperature::Int, fish::Int)
    return matread(artr_wolf_2023_distances_file(; temperature, fish))["dist_"]
end

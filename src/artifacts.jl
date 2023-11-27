#= Dataset from Le Goc et al 2021, as prepared by Matteo. =#
function behaviour_free_swimming_hdf5_path()
    return joinpath(artifact"behaviour_free_swimming", "behaviour_free_swimming.h5")
end

function behaviour_free_swimming_temperatures()
    return (18, 22, 26, 30, 33)
end

function load_behaviour_free_swimming_data(temperature::Int)
    if temperature ∉ (18, 22, 26, 30, 33)
        throw(ArgumentError("temperature must be one of 18, 22, 26, 30, 33; got $temperature"))
    end

    dataset_names = ("bouttime", "displacements", "dtheta", "interboutintervals", "xpos", "ypos")

    h5open(behaviour_free_swimming_hdf5_path()) do h5
        group = h5["behaviour/$temperature"]

        datasets = map(dataset_names) do dataset
            read(group, dataset)
        end

        temperature = attrs(group)["temperature"]
        units = Dict(dataset => attrs(group[dataset])["unit"] for dataset = dataset_names)

        return (; zip(Symbol.(dataset_names), datasets)..., temperature, units)
    end
end

#= Original dataset from Le Goc et al 2021, which should contain long trajectories =#
function legoc2021_data_dir()
    return artifact"LeGoc2021"
end

function legoc2021_single_fish_T26_path()
    return joinpath(legoc2021_data_dir(), "sf_allsequences_T26.mat")
end

function wolf_eyes_data()
    mat = matread(joinpath(wolf_eyes_data_dir(), "analysis.mat"))
    return mat["D"]
end
function wolf_eyes_data()
    mat = matread(joinpath(wolf_eyes_data_dir(), "analysis.mat"))
    return mat["D"]
end

function wolf_eyes_artr_data_dir()
    return artifact"Wolf_Eyes_ARTR_Data_20240313"
end

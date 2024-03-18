function wolf_eyes_data()
    mat = matread(joinpath(wolf_eyes_data_dir(), "analysis.mat"))
    return mat["D"]
end

function
    artifact"Wolf_Eyes_ARTR_Data_20240313"
end

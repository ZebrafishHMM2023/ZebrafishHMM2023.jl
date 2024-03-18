function wolf_eyes_data()
    mat = matread(joinpath(wolf_eyes_data_dir(), "analysis.mat"))
    return mat["D"]
end

#= Neural recordings corresponding to the eye gaze data =#
function wolf_eyes_artr_data_dir()
    return artifact"Wolf_Eyes_ARTR_Data_20240313"
end

function wolf_eyes_artr_data()
    left = matread(joinpath(wolf_eyes_artr_data_dir(), "A_Left_ARTR.mat"))["A_Left_ARTR"]
    right = matread(joinpath(wolf_eyes_artr_data_dir(), "A_Right_ARTR.mat"))["A_Right_ARTR"]
    corr_left = matread(joinpath(wolf_eyes_artr_data_dir(), "DFF_corr_Left_ARTR.mat"))["DFF_corr_Left_ARTR"]
    corr_right = matread(joinpath(wolf_eyes_artr_data_dir(), "DFF_corr_Right_ARTR.mat"))["DFF_corr_Right_ARTR"]
    return (; left, right, corr_left, corr_right)
end

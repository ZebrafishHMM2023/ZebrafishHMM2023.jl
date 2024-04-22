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

#= Second eye movement data, 20240415, from S.Wolf =#
function wolf_eyes_20240415_data_dir()
    return artifact"Wolf_Eyes_ARTR_Data_20240415"
end

function wolf_eyes_20240415_data()
    left = matread(joinpath(wolf_eyes_20240415_data_dir(), "2014-12-19 Run 02", "ARTR neural signals", "A_Left_ARTR.mat"))["A_Left_ARTR"]
    right = matread(joinpath(wolf_eyes_20240415_data_dir(), "2014-12-19 Run 02", "ARTR neural signals", "A_Right_ARTR.mat"))["A_Right_ARTR"]
    corr_left = matread(joinpath(wolf_eyes_20240415_data_dir(), "2014-12-19 Run 02", "ARTR neural signals", "DFF_corr_Left_ARTR.mat"))["DFF_corr_Left_ARTR"]
    corr_right = matread(joinpath(wolf_eyes_20240415_data_dir(), "2014-12-19 Run 02", "ARTR neural signals", "DFF_corr_Right_ARTR.mat"))["DFF_corr_Right_ARTR"]

    D_struct = matread(joinpath(wolf_eyes_20240415_data_dir(), "2014-12-19 Run 02", "2014-12-19-run2.mat"))["D"]
    position = D_struct["position"]
    timeplusleft = D_struct["timeplusleft"]
    timeplusright = D_struct["timeplusright"]
    timeminusleft = D_struct["timeminusleft"]
    timeminusright = D_struct["timeminusright"]

    return (; left, right, corr_left, corr_right, position, timeplusleft, timeplusright, timeminusleft, timeminusright)
end

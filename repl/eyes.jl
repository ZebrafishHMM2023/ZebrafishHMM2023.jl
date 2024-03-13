import CairoMakie
import Makie
import MAT
import ZebrafishHMM2023

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=700, height=200, xlabel="time", ylabel="gaze orientation")
    Makie.lines!(
        ax,
        vec(MAT.matread(joinpath(ZebrafishHMM2023.wolf_eyes_data_dir(), "analysis.mat"))["D"]["gaze"]["orient"])
    )

    ax = Makie.Axis(fig[2,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    Makie.hist!(
        ax,
        vec(MAT.matread(joinpath(ZebrafishHMM2023.wolf_eyes_data_dir(), "analysis.mat"))["D"]["gaze"]["orient"])
    )

    Makie.resize_to_layout!(fig)
    fig
end

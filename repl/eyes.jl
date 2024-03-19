import CairoMakie
import Makie
import MAT
import ZebrafishHMM2023

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=700, height=200, xlabel="time", ylabel="gaze orientation")
    Makie.lines!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=400, height=400, xlabel="gaze orientation", ylabel="count")
    Makie.hist!(ax, vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"]))
    Makie.resize_to_layout!(fig)
    fig
end


ZebrafishHMM2023.wolf_eyes_artr_data().left
vec(ZebrafishHMM2023.wolf_eyes_data()["gaze"]["orient"])


ZebrafishHMM2023.wolf_eyes_artr_data().left


diff(vec(ZebrafishHMM2023.wolf_eyes_data()["tframes"]))


ZebrafishHMM2023.wolf_eyes_data()["tframes"]

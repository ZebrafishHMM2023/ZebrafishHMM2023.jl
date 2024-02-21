using ZebrafishHMM2023: load_behaviour_free_swimming_data

data = load_behaviour_free_swimming_data(18)

interboutintervals_trajs = [filter(!isnan, traj) for traj = eachcol(data.interboutintervals)]
displacements_trajs = [filter(!isnan, traj) for traj = eachcol(data.displacements)]

using GZip, CSV, DataFrames, Printf, LowRankModels, Tables

# Fit low rank models to the BMT and Onc data.

include("factor_setup.jl")

function getobs(x)
    obs = Tuple{Int,Int}[]
    for i = 1:size(x, 1)
        for j = 1:size(x, 2)
            if !ismissing(x[i, j])
                push!(obs, (i, j))
            end
        end
    end
    return obs
end

for src in ["bmt", "onc"]

    mx, dyad_info = factor_setup(src, save = true)
    continue
    obs = getobs(mx)

    # Fit 1, 2, and 3 factor models.
    for d = 1:3
        m = GLRM(mx, QuadLoss(), ZeroReg(), ZeroReg(), d, obs = obs)
        r = fit!(m)
        GZip.open("$(src)_$(d)_u.csv.gz", "w") do io
            CSV.write(io, Tables.table(r[1]'))
        end
        GZip.open("$(src)_$(d)_v.csv.gz", "w") do io
            CSV.write(io, Tables.table(r[2]'))
        end
    end
end

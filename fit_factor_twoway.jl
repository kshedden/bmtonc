using CompositeMultivariateAnalysis
using Statistics
using Random
using LinearAlgebra
using Serialization

dp = "/home/kshedden/data/Sung_Choi/wide"

include("utils.jl")

function write_ss_info(out, da, dm)
    write(out, @sprintf("%d distinct dyads\n", length(unique(da[1][:, :ID]))))
    write(out, @sprintf("%d dyad-days\n", size(da[1], 1)))

    b = 100 * (1 - mean(ismissing.(dm[1])))
    write(out, @sprintf("%d%% of potential patient values are observed\n", round(b)))
    b = 100 * (1 - mean(ismissing.(dm[2])))
    write(out, @sprintf("%d%% of potential caregiver values are observed\n\n", round(b)))
end

function rand_ix(da, mode)

    if mode == 1
        # Simple randomization
        n = size(da[1], 1)
        return shuffle(1:n)
    elseif mode == 2
        # Randomize days within people
        d2 = copy(da[2])
        d2[:, :ix] = 1:size(d2, 1)
        for dd in groupby(d2, :ID)
            shuffle!(dd)
        end
        return d2[:, :ix]
    end
end

function randomize(da, dm, d, cfg; mode=1, nrep=10)

    d1 = copy(dm[1])
    d2 = copy(dm[2])

    cx = []
    for r in 1:nrep
        ii = rand_ix(da, mode)
        d2 = d2[ii, :]
        rr = fit(BiMVA, d1, d2; d=d, config=cfg, verbose=true)
        CompositeMultivariateAnalysis.rotate!(rr)
        cr = CompositeMultivariateAnalysis.cor(rr)
        push!(cx, cr)
    end

    return copy(hcat(cx...)')
end

function splitsample(d, out; nrep=100)

    cfg = BiMVAconfig(cca=1., pcax=0.5, pcay=0.5)

    id = sort(unique(da[1][:, :ID]))
    n = length(id)
    n2 = Int(round(0.8*n))

    cra = []
    for j in 1:nrep
        shuffle!(id)
        id1 = id[1:n2]
        id2 = id[n2+1:end]
        m = size(dm[1], 1)
        jj1 = [x in id1 for x in da[1][:, :ID]]
        jj2 = [x in id2 for x in da[1][:, :ID]]
        rr = fit(BiMVA, dm[1][jj1, :], dm[2][jj1, :]; d=d, config=cfg, verbose=false)
        rx = fit(BiMVA, dm[1][jj2, :], dm[2][jj2, :]; d=d, config=cfg, verbose=false, dofit=false)
        CompositeMultivariateAnalysis.rotate!(rr)
        A,B = coef(rr)
        cr = diag(A'*rx.Sxy*B) ./ (sqrt.(diag(A'*rx.Sxx*A)) .* sqrt.(diag(B'*rx.Syy*B)))
        push!(cra, cr)
    end
    cra = hcat(cra...)

    write(out, "Split sample analysis:\n")
    for p in [0.025, 0.975]
        write(out, @sprintf("p=%.3f\n", p))
        show(out, "text/plain", [quantile(z, p) for z in eachrow(cra)])
        write(out, "\n")
    end
end

d = 5
da, dm = load_data2()

out = open("2way_results.txt", "w")

write_ss_info(out, da, dm)

cra = splitsample(d, out; nrep=100)

cfg = BiMVAconfig(cca=1., pcax=0.5, pcay=0.5)

rr = fit(BiMVA, dm[1], dm[2]; d=d, config=cfg, verbose=true)
CompositeMultivariateAnalysis.rotate!(rr)

open(joinpath(dp, "rr.ser"), "w") do io
    serialize(io, rr)
end

scores_pt, scores_cg = predict(rr; verbose=false)
open(joinpath(dp, "scores_pt.ser"), "w") do io
    serialize(io, scores_pt)
end
open(joinpath(dp, "scores_cg.ser"), "w") do io
    serialize(io, scores_cg)
end

cr = CompositeMultivariateAnalysis.cor(rr)
write(out, "\nCorrelations:\n")
show(out, "text/plain", cr)

cx1 = randomize(da, dm, d, cfg; mode=1, nrep=5)
write(out, "\n\nRandomized correlations (randomize observations):\n")
show(out, "text/plain", cx1)

cx2 = randomize(da, dm, d, cfg; mode=2, nrep=5)
write(out, "\n\nRandomized correlations (randomize subjects):\n")
show(out, "text/plain", cx2)

close(out)

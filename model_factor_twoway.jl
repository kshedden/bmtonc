using LinearAlgebra
using CSV
using Dates
using CompositeMultivariateAnalysis
using DataFrames
using Printf
using Statistics
using StatsBase
using Random
using CairoMakie
using Serialization
using MixedModels
using EstimatingEquationsRegression
using GLM
using UnicodePlots

rm("plots", recursive = true, force = true)
mkdir("plots")

include("utils.jl")
include("model_utils.jl")

dp = "/home/kshedden/data/Sung_Choi/wide"

demog = load_demog2()
mood = load_mood()

function plot_factors1(rr, jc, Zm, Zc, scores, loadings, grp, cr, ifig)

    (; Sxx, Syy, A, B) = rr

    zcov = grp == "patient" ? Sxx : Syy
    pve = (loadings' * zcov * loadings) / maximum(eigen(zcov).values)

    n = size(loadings, 1)
    tix = range(0, 24, n)

    f = Figure()
    title = @sprintf("%s component %d (r=%.2f, PVE=%.2f)", titlecase(grp), jc, cr, pve)
    ax = Axis(f[1, 1], xlabel="Time", ylabel="Heartrate", title=title, xticks=0:3:24,
              xlabelsize=18, ylabelsize=18)

    lines!(tix, Zm, color=:black)
    s = std(skipmissing(scores))
    lines!(tix, Zm + s*loadings, color=:red)
    lines!(tix, Zm + 2*s*loadings, color=:red, linestyle=:dot)
    lines!(tix, Zm - s*loadings, color=:blue)
    lines!(tix, Zm - 2*s*loadings, color=:blue, linestyle=:dot)

    save(@sprintf("plots/%03d.pdf", ifig), f)

    return ifig + 1
end

function plot_factors(rr, cr, ifig)

    (; Xm, Xc, Ym, Yc, Sxx, A, d) = rr

    A, B = coef(rr)

    for j in 1:d
        ifig = plot_factors1(rr, j, Xm, Xc, scores_pt[:, j], A[:, j], "patient", cr[j], ifig)
        ifig = plot_factors1(rr, j, Ym, Yc, scores_cg[:, j], B[:, j], "caregiver", cr[j], ifig)
    end

    return ifig
end


function profile_scores1(dx, mstruct, out)

    fml, dx = setup_dx(mstruct, true, dx)

    contrasts = Dict(:dayofweek=>DummyCoding())
    mm = fit(MixedModel, fml, dx, contrasts=contrasts)
    write(out, @sprintf("\n%d distinct people\n", length(unique(dx[:, :person]))))
    write(out, @sprintf("%d distinct dyads\n", length(unique(dx[:, :dyad]))))
    write(out, @sprintf("%d distinct dyad-days\n", length(unique(dx[:, :dyadday]))))
    ss = mm.σs
    v_dyadday = first(ss.dyadday)^2
    v_person = first(ss.person)^2
    v_dyad = first(ss.dyad)^2
    v_resid = sdest(mm)^2
    v_total = v_dyadday + v_person + v_dyad + v_resid
    write(out, @sprintf("Proportion of dyad variance: %f\n", v_dyad / v_total))
    write(out, @sprintf("Proportion of dyad-day variance: %f\n", v_dyadday / v_total))
    write(out, string(mm))
    write(out, "\n")

    if mstruct == 1
        amp, peak = get_season(mm)
        write(out, @sprintf("Seasonal amplitude: %.2f\n", amp))
        write(out, @sprintf("Seasonal peak: %.2f\n\n", peak))
    end

    return mm
end

function profile_scores_gee1(dx, mstruct, out)

    fml, dx = setup_dx(mstruct, false, dx)

    contrasts = Dict(:dayofweek=>DummyCoding())
    mm = gee(fml, dx, dx[:, :dyad], IdentityLink(), ConstantVar(),
             IndependenceCor(); bccor=false, contrasts=contrasts)
    write(out, @sprintf("\n%d distinct people\n", length(unique(dx[:, :person]))))
    write(out, @sprintf("%d distinct dyads\n", length(unique(dx[:, :dyad]))))
    write(out, @sprintf("%d distinct dyad-days\n", length(unique(dx[:, :dyadday]))))
    write(out, string(mm))
    write(out, "\n")

    if mstruct == 1
        amp, peak = get_season(mm)
        write(out, @sprintf("Seasonal amplitude: %.2f\n", amp))
        write(out, @sprintf("Seasonal peak: %.2f\n\n", peak))
    end

    scoef = coef(mm) .* std(modelmatrix(mm), dims=1)[:]

    return mm, scoef
end

function profile_scores(scores_pt, scores_cg, out)

    d = size(scores_pt, 2)
    scoef = [[], [], []]
    names = [[], [], []]
    for j in 1:d

        dx = prep_profile(scores_pt[:, j], scores_cg[:, j], demog, mood, da)

        write(out, @sprintf("\n\nComponent %d:\n", j))
        for mstruct in 1:3
            #mmix = profile_scores1(dx, mstruct, out)
            mgee, scoef1 = profile_scores_gee1(dx, mstruct, out)
            push!(scoef[mstruct], scoef1)
            names[mstruct] = coefnames(mgee)
        end
    end

    return scoef, names
end


function sample_scores(scores, dm, group, ifig)

    n = size(dm, 2)
    @assert size(dm, 1) == size(scores, 1)
    tix = range(0, 24, n)

    for j in 1:size(scores, 2)

        x = zeros(n)
        for i in 1:n
            for k in 1:size(scores, 2)
                if ismissing(scores[i, k])
                    continue
                end
                if k == j
                    x[i] += abs(scores[i, k])
                else
                    x[i] -= abs(scores[i, k])/size(scores, 2)
                end
            end
        end

        ii = sortperm(x; rev=true)
        f = Figure()
        title = "Raw $(group) data scoring mainly on component $(j)"
        ax = Axis(f[1, 1], xlabel="Time", ylabel="Heartrate", title=title, xticks=0:3:24,
                  xlabelsize=18, ylabelsize=18)
        npos, nneg = 0, 0
        for i in ii
            if scores[i, j] > 0 && npos < 10
                lines!(tix, dm[i, :]; color=(:red,0.5))
                npos += 1
            elseif scores[i, j] < 0 && nneg < 10
                lines!(tix, dm[i, :]; color=(:blue,0.5))
                nneg += 1
            end
            if min(nneg, npos) >= 10
                break
            end
        end

        save(@sprintf("plots/%03d.pdf", ifig), f)
        ifig += 1
    end

    return ifig
end

function biplot_coef(scoef, vnames, ifig)

    ii = findall(x->!occursin("intercept", lowercase(x)), vnames)
    cf = hcat(scoef...)
    cf = cf[ii, :]
    vnames = vnames[ii]

    dr = DataFrame(name=vnames)
    for j in 1:size(cf, 2)
        dr[:, string(j)] = cf[:, j]
    end

    ii = findfirst(x->occursin("intervention", lowercase(x)), vnames)
    vnames[ii] = "intervention"

    u,s,v = svd(cf)
    s = s.^2
    s ./= sum(s)
    println(s)

    vnames = copy(vnames)
    vnames = [replace(x, "dayofweek: "=>"") for x in vnames]
    vnames = [replace(x, "cg_relationship: "=>"") for x in vnames]
    vnames = [replace(x, "cg_arm: "=>"") for x in vnames]
    vnames = [replace(x, "role: "=>"") for x in vnames]
    vnames = [replace(x, "gender: "=>"") for x in vnames]

    f = Figure()
    title = @sprintf("Coefficient biplot")
    ax = Axis(f[1, 1], xlabel="Dimension 1", ylabel="Dimension 2", title=title,
              xlabelsize=18, ylabelsize=18)

    for i in 1:size(v,1)
        lines!(ax, [v[i, 1], -v[i, 1]], [v[i, 2], -v[i, 2]], color=:grey, align=:center)
    end

    text!(ax, u[:, 1], u[:, 2], text=vnames)
    text!(ax, v[:, 1], v[:, 2], text=string.(1:size(v,1)))

    save(@sprintf("plots/%03d.pdf", ifig), f)
    ifig += 1

    return dr, ifig
end

function make_biplots(ifig)
    for k in 1:3
        dr, ifig = biplot_coef(scoef[k], vnames[k], ifig)
        CSV.write("coef_std$(k).csv", dr)
    end
    return ifig
end


rr = open(joinpath(dp, "rr.ser")) do io
    deserialize(io)
end

scores_pt = open(joinpath(dp, "scores_pt.ser")) do io
    deserialize(io)
end

scores_cg = open(joinpath(dp, "scores_cg.ser")) do io
    deserialize(io)
end

cr = CompositeMultivariateAnalysis.cor(rr)
da, dm = load_data2()

out = open("2way_models.txt", "w")

ifig = 0
ifig = plot_means(rr, ifig)
ifig = plot_factors(rr, cr, ifig)

ifig = sample_scores(scores_pt, dm[1], "patient", ifig)
ifig = sample_scores(scores_cg, dm[2], "caregiver", ifig)

scoef, vnames = profile_scores(scores_pt, scores_cg, out)

ifig = make_biplots(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=$2way.pdf $f`
run(c)

close(out)

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

function prep_profile(scores_pt, scores_cg, demog, mood)

    id = vcat(da[1][:, :ID], da[2][:, :ID])
    scores = vcat(scores_pt, scores_cg)
    date = vcat(da[1][:, :Date], da[2][:, :Date])
    day = vcat(da[1][:, :day], da[2][:, :day])
    n = size(da[1], 1)
    dyad = ["$x-$y" for (x, y) in zip(da[1][:, :ID], da[2][:, :ID])]
    dyad = vcat(dyad, dyad)
    dx = DataFrame(person=id, scores=scores, date=date, dyad=dyad, day=day)
    dx[:, :dayofweek] = [ismissing(x) ? missing : dayofweek(x) for x in dx[:, :date]]
    dx[:, :dayofyear] = [ismissing(x) ? missing : dayofyear(x) for x in dx[:, :date]]
    dx[:, :season_cos] = cos.(2*pi*dx[:, :dayofyear] / 365.25)
    dx[:, :season_sin] = sin.(2*pi*dx[:, :dayofyear] / 365.25)
    dd = Dict(1=>"Mo", 2=>"Tu", 3=>"We", 4=>"Th", 5=>"Fr", 6=>"Sa", 7=>"Su", missing=>missing)
    dx[!, :dayofweek] = [dd[x] for x in dx[:, :dayofweek]]
    dx[:, :datecat] = [ismissing(x) ? missing : string(x) for x in dx[:, :date]]
    dx[:, :dyadday] = [ismissing(x) || ismissing(y) ? missing : @sprintf("%s:%s", x, y) for
                       (x, y) in zip(dx[:, :datecat], dx[:, :dyad])]

    # Merge in the demographic variables
    dx = leftjoin(dx, demog, on=:person=>:ID)

    # Merge in the mood variables
    mood = select(mood, Not(:role))
    dx = leftjoin(dx, mood, on=[:person=>:id, :date=>:date])

    dx = sort(dx, [:dyad, :person, :date])

    return dx
end

function setup_dx(mstruct, ranef, dx)

    v = if mstruct == 1
        [:role, :gender, :dayofweek, :season_cos, :season_sin, :age,
         :cg_relationship, :cg_arm, :day]
    elseif mstruct == 2
        [:role, :gender, :dayofweek, :season_cos, :season_sin, :age,
         :cg_relationship, :cg_arm, :day, :mood1]
    else
        error("Unknown mstruct")
    end

    f = if ranef
        term(:scores) ~ term(1) + sum(term.(v)) + (term(1) | term(:dyad)) + (term(1) | term(:person)) + (term(1) | term(:dyadday))
    else
        term(:scores) ~ term(1) + sum(term.(v))
    end

    v = vcat(:scores, v, :dyad, :person, :dyadday)
    dx = dx[:, v]
    dx = dx[completecases(dx), :]

    return f, dx
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

    return mm
end

function profile_scores(scores_pt, scores_cg, out)

    d = size(scores_pt, 2)
    for j in 1:d

        dx = prep_profile(scores_pt[:, j], scores_cg[:, j], demog, mood)

        write(out, @sprintf("\n\nComponent %d:\n", j))
        for mstruct in 1:2
            #mmix = profile_scores1(dx, mstruct, out)
            mgee = profile_scores_gee1(dx, mstruct, out)
        end
    end
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

profile_scores(scores_pt, scores_cg, out)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=$2way.pdf $f`
run(c)

close(out)

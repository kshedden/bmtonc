using CodecZlib, CSV, Tables, LinearAlgebra, Statistics, PyPlot
using UMAP, Printf, Distributions, Random

Random.seed!(42)

rm("plots", recursive = true, force = true)
mkdir("plots")

include("factor_setup.jl")

cols = ["blue", "lightblue", "grey", "orange", "red"]

# Center the first half of the rows of u (caregivers) and the second
# half of the rows of u (patients).
function center_subjects(u::AbstractMatrix)
    u = copy(u)
    n = size(u, 1)
    m = div(n, 2)

    for j = 1:size(u, 2)
        u[1:m, j] .-= mean(u[1:m, j])
        u[m+1:end, j] .-= mean(u[m+1:end, j])
    end

    return u
end

function vardecomp(v::AbstractVector, dyad_info::AbstractDataFrame, randomize::Bool)

    # i=dyad, j=person, k=day
    # y_ijk = a_i + b_ij + c_ik + e_ijk 

    y = (v .- mean(v)) / std(v)

    dyad_info_cg = copy(dyad_info)
    dyad_info_cg[:, :role] .= "caregiver"
    dyad_info_pt = copy(dyad_info)
    dyad_info_pt[:, :role] .= "patient"

    di = if randomize
        ii = sortperm(rand(size(dyad_info, 1)))
        vcat(dyad_info_cg, dyad_info_pt[ii, :])
    else
        vcat(dyad_info_cg, dyad_info_pt)
    end

    di[:, :y] = y

    n = length(v)
    m = div(n, 2)

    # Same dyad
    ss1, qq1 = 0.0, 0

    # Same dyad/person
    ss2, qq2 = 0.0, 0

    # Same dyad/day
    ss3, qq3 = 0.0, 0

    # Same person/day
    ss4, qq4 = 0.0, 0

    for d1 in groupby(di, :ID)

        n = size(d1, 1)
        m = div(n, 2)

        for i1 = 1:m
            # Var(c_ik) -- same day different people
            #@assert d1[i1, :Day] == d1[i1+m, :Day]
            ss3 += d1[i1, :y] * d1[i1+m, :y]
            qq3 += 1

            # Same day same person
            ss4 += d1[i1, :y]^2
            ss4 += d1[i1+m, :y]^2
            qq4 += 2

            for i2 = 1:m
                if i1 != i2
                    # Var(a_i) -- different days different people
                    #@assert d1[i1, :Day] != d1[i2+m, :Day]
                    #@assert d1[i1, :role] != d1[i2+m, :role]
                    ss1 += d1[i1, :y] * d1[i2+m, :y]
                    qq1 += 1

                    # Var(b_ij) -- different day same person
                    #@assert d1[i1, :role] == d1[i2, :role]
                    #@assert d1[i1, :Day] != d1[i2, :Day]
                    ss2 += d1[i1, :y] * d1[i2, :y]
                    #@assert d1[i1+m, :role] == d1[i2+m, :role]
                    #@assert d1[i1+m, :Day] != d1[i2+m, :Day]
                    ss2 += d1[i1+m, :y] * d1[i2+m, :y]
                    qq2 += 2
                end
            end
        end
    end

    ss1 = clamp(ss1 / qq1, 0, Inf)
    ss2 = clamp(ss2 / qq2, 0, Inf)
    ss3 = clamp(ss3 / qq3, 0, Inf)
    ss4 = clamp(ss4 / qq4, 0, Inf)
    x = [ss1, ss2, ss3, ss4]
    x ./= sum(x)
    return x
end

# Rotate so that the scores for different factors are orthogonal.
# 'u' contains the looadings and 'v' contains the scores.
function rotate_orthog(u::AbstractMatrix, v::AbstractMatrix)

    # Center the scores
    mn = mean(v, dims = 1)[:]
    for j = 1:size(v, 2)
        v[:, j] .-= mn[j]
    end

    # Handle the one factor case separately.
    if size(u, 2) == 1
        f = norm(u)
        return u ./ f, v .* f, mn
    end

    u1, s1, v1 = svd(v)
    u = u * v1 * diagm(s1)
    v = u1

    # The loading vectors should have unit norm.
    for j = 1:size(u, 2)
        f = norm(u[:, j])
        u[:, j] ./= f
        v[:, j] .*= f
    end

    return u, v, mn
end

function xfill(uu::AbstractMatrix, vv::AbstractMatrix, np::Int, dyad_info)

    n, q = size(uu)
    @assert size(uu, 2) == size(vv, 2)

    loadall = zeros(1440, np)
    vdb = Float64[0, 0, 0, 0]
    vdc = Any[nothing, nothing, nothing, nothing]
    vds = Any[nothing, nothing, nothing, nothing]
    vdm = []
    for ii = 1:np
        println("ii=", ii)
        @assert isapprox(mean(vv, dims = 1)[:], zeros(size(vv, 2)), atol = 1e-8)

        # Generate a random rotation of the low rank fit.
        dv = diagm(1 ./ sqrt.(diag(vv' * vv / n)))
        dt = randn(q, q)
        dt, _, _ = svd(dt)
        dt = dv * dt
        uu = uu / dt'  # loadings
        vv = vv * dt # scores
        for j = 1:size(vv, 2)
            f = norm(uu[:, j])
            uu[:, j] ./= f
            vv[:, j] .*= f

            # Flip the loadings so that more are positive than negative
            if sum(uu[:, j] .< 0) > sum(uu[:, j] .>= 0)
                uu[:, j] .*= -1
                vv[:, j] .*= -1
            end
        end

        vvx = center_subjects(vv)[:, 1]
        vd = vardecomp(vvx, dyad_info, false)
        push!(vdm, vd)
        loadall[:, ii] = uu[:, 1]

        for j = 1:4
            if ii == 1 || vd[j] > vdb[j]
                vdb[j] = vd[j]
                vdc[j] = uu[:, 1]
                vds[j] = vvx
            end
        end
    end

    return hcat(vdm...), vdb, vdc, vds, loadall
end

function plot_mean(src, cg_mean, pt_mean, ifig)
	hh = collect(range(1, 1440) / 60)
	PyPlot.clf()
	PyPlot.axes([0.12, 0.12, 0.68, 0.8])
	PyPlot.grid(true)
	PyPlot.plot(hh, cg_mean[:, :hrtrt], "-", label="Caregiver", color="orange")
	PyPlot.plot(hh, pt_mean[:, :hrtrt], "-", label="Patient", color="purple")
	PyPlot.title(uppercase(src))
    PyPlot.xlabel("Hour of day", size = 15)
    PyPlot.ylabel("Mean heart rate", size = 15)
    PyPlot.xlim(0, 24)
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
	leg.draw_frame(false)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end

function plot_components(src, uu, vv, cg_mean, pt_mean, ifig)
    n = div(size(uu, 1), 2)
    hh = collect(range(1, 1440) / 60)
    for j = 1:size(uu, 2)
        PyPlot.clf()
        PyPlot.grid(true)
        for (jj, p) in enumerate([0.2, 0.35, 0.5, 0.65, 0.8])
            s = quantile(vv[1:n, j], p)
            PyPlot.plot(hh, cg_mean[:, :hrtrt] + s * uu[:, j], "-", color = cols[jj])
        end
        PyPlot.title(
            @sprintf("%s fitted means for caregivers (component %d)", uppercase(src), j)
        )
        PyPlot.xlabel("Hour of day", size = 15)
        PyPlot.ylabel("Heart rate", size = 15)
        PyPlot.xlim(0, 24)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        PyPlot.clf()
        PyPlot.grid(true)
        for (jj, p) in enumerate([0.2, 0.35, 0.5, 0.65, 0.8])
            s = quantile(vv[n+1:end, j], p)
            PyPlot.plot(hh, pt_mean[:, :hrtrt] + s * uu[:, j], "-", color = cols[jj])
        end
        PyPlot.title(
            @sprintf("%s fitted means for patients (component %d)", uppercase(src), j)
        )
        PyPlot.xlabel("Hour of day", size = 15)
        PyPlot.ylabel("Heart rate", size = 15)
        PyPlot.xlim(0, 24)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end
    return ifig
end

function save_umap(loadall::AbstractMatrix, um::AbstractMatrix, src::String, qq::Int)
	f = @sprintf("%s_%d_loadall.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(loadall))
	end
	f = @sprintf("%s_%d_um.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(um))
	end
	println(size(um))
	println(size(loadall))
end

function submain(src, qq, out, ifig)

    mx, dyad_info = factor_setup(src)
	write(out, @sprintf("%s %d total person/days\n", uppercase(src),
	      size(mx, 1)))

    cg_mean = open(GzipDecompressorStream, "cg_mean_$(src).csv.gz") do io
        CSV.read(io, DataFrame)
    end

    pt_mean = open(GzipDecompressorStream, "pt_mean_$(src).csv.gz") do io
        CSV.File(io) |> DataFrame
    end

	ifig = plot_mean(src, cg_mean, pt_mean, ifig)

    # Loadings
    uu = open(GzipDecompressorStream, "$(src)_$(qq)_u.csv.gz") do io
        CSV.File(io) |> Tables.matrix
    end

    # Scores
    vv = open(GzipDecompressorStream, "$(src)_$(qq)_v.csv.gz") do io
        CSV.File(io) |> Tables.matrix
    end

    uu, vv, mn = rotate_orthog(uu, vv)
    ifig = plot_components(src, uu, vv, cg_mean, pt_mean, ifig)
    cc = vv + ones(size(vv, 1)) * mn'
    fit = uu * cc'
    pve = 1 - sum(abs2, skipmissing(fit - mx)) / sum(abs2, skipmissing(mx))
    write(out, @sprintf("%s combined factor PVE: %f\n", uppercase(src), pve))

    vdm, vdb, vdc, vds, loadall = xfill(uu, vv, 2000, dyad_info)
	xx = vcat(vdm, mean(loadall, dims=1))
    um = umap(xx, 2)

	save_umap(loadall, um, src, qq)

    for k = 1:4
        fit = vdc[k] * (vds[k] .+ mn[k])'
        pve = 1 - sum(abs2, skipmissing(fit - mx)) / sum(abs2, skipmissing(mx))
        a = ["dyad", "person", "dyad day", "residual"][k]
        write(out, @sprintf("%s %s PVE: %f\n", uppercase(src), a, pve))
        write(out, @sprintf("%s %s ICC: %f\n", uppercase(src), a, vdb[k]))
    end

    di = vcat(dyad_info, dyad_info)
    @assert size(di, 1) == size(vds[1], 1)

    # Colot the points using a scalar summary of the loading
    # pattern shape.
    PyPlot.clf()
    PyPlot.grid(true)
    c = mean(loadall, dims = 1)[:]
    PyPlot.scatter(um[1, :], um[2, :], c = c, cmap = "cool", alpha = 0.4, rasterized=true)
    PyPlot.title(@sprintf("%s variance component loading pattern", uppercase(src)))
    PyPlot.xlabel("UMAP component 1", size = 15)
    PyPlot.ylabel("UMAP component 2", size = 15)
    PyPlot.colorbar()
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    for k = 1:4

        # Plot the UMAP embedding of the ICC values
        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.scatter(um[1, :], um[2, :], c = vdm[k, :], cmap = "cool", alpha = 0.4, rasterized=true)
        PyPlot.title(
            @sprintf(
                "%s %s ICC",
                uppercase(src),
                ["dyad", "person", "dyad day", "residual"][k]
            )
        )
        PyPlot.xlabel("UMAP component 1", size = 15)
        PyPlot.ylabel("UMAP component 2", size = 15)
        PyPlot.colorbar()
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        # Plot the loadings
        PyPlot.clf()
        PyPlot.axes([0.15, 0.12, 0.8, 0.8])
        PyPlot.grid(true)
        hh = collect(range(1, 1440) / 60)
        PyPlot.plot(hh, vdc[k], "-", color = "orange")
        if minimum(vdc[k]) > 0
            PyPlot.ylim(ymin = 0)
        elseif maximum(vdc[k]) < 0
            PyPlot.ylim(ymax = 0)
        end
        PyPlot.title(uppercase(src))
        PyPlot.xlabel("Hour of day", size = 15)
        a = ["Dyad", "Person", "Dyad day", "Residual"][k]
        PyPlot.ylabel("$(a) variance component loading", size = 15)
        PyPlot.xlim(0, 24)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        n = div(size(vds[k], 1), 2)

        # Plot mean + loadings for caregivers
        PyPlot.clf()
        PyPlot.grid(true)
        for (jj, p) in enumerate([0.2, 0.35, 0.5, 0.65, 0.8])
            s = quantile(vds[k][1:n], p)
            PyPlot.plot(hh, cg_mean[:, :hrtrt] + s * vdc[k], "-", color = cols[jj])
        end
        PyPlot.title(
            @sprintf(
                "%s fitted means for caregivers (%s variance component)",
                uppercase(src),
                ["dyad", "person", "dyad day", "residual"][k]
            )
        )
        PyPlot.xlabel("Hour of day", size = 15)
        PyPlot.ylabel("Heart rate", size = 15)
        PyPlot.xlim(0, 24)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        # Plot mean + loadings for patients
        PyPlot.clf()
        PyPlot.grid(true)
        for (jj, p) in enumerate([0.2, 0.35, 0.5, 0.65, 0.8])
            s = quantile(vds[k][n+1:end], p)
            PyPlot.plot(hh, pt_mean[:, :hrtrt] + s * vdc[k], "-", color = cols[jj])
        end
        PyPlot.title(
            @sprintf(
                "%s fitted means for patients (%s variance component)",
                uppercase(src),
                ["dyad", "person", "dyad day", "residual"][k]
            )
        )
        PyPlot.xlabel("Hour of day", size = 15)
        PyPlot.ylabel("Heart rate", size = 15)
        PyPlot.xlim(0, 24)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end
    return ifig
end

# Use the solutions with qq factors
qq = 5

function main(ifig)

    out = open("factor_map_$(qq).txt", "w")

    for src in ["bmt", "onc"]
        ifig = submain(src, qq, out, ifig)
    end

    close(out)

    return ifig
end

ifig = 0
ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=factor_map_$(qq).pdf $f`
run(c)

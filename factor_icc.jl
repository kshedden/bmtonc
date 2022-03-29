using CodecZlib, CSV, Tables, LinearAlgebra, Statistics, PyPlot, Printf

rm("plots", recursive = true, force = true)
mkdir("plots")

include("factor_setup.jl")

src = "bmt"
mx, dyad_info = factor_setup(src)

# Center the first half of the rows of u (caregivers) and the second
# half of the rows of u (patients).
function center_subjects(u)
    u = copy(u)
    n = size(u, 1)
    m = div(n, 2)

    for j = 1:size(u, 2)
        u[1:m, j] .-= mean(u[1:m, j])
        u[m+1:end, j] .-= mean(u[m+1:end, j])
    end

    return u
end

function vardecomp(v::AbstractVector, dyad_info, randomize::Bool)

    # i=dyad, j=person, k=day
    # y_ijk = a_i + b_ij + c_ik + e_ijk 

    y = (v .- mean(v)) / std(v)

    di = if randomize
        ii = sortperm(rand(size(dyad_info, 1)))
        vcat(dyad_info, dyad_info[ii, :])
    else
        vcat(dyad_info, dyad_info)
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

    for d1 in groupby(di, :ID)

        n = size(d1, 1)
        m = div(n, 2)

        for i1 = 1:m
            # Var(c_ik) -- same day different people
            ss3 += d1[i1, :y] * d1[i1+m, :y]
            qq3 += 1

            for i2 = 1:m
                if i1 != i2
                    # Var(a_i) -- different days different people
                    ss1 += d1[i1, :y] * d1[i2+m, :y]
                    qq1 += 1

                    # Var(b_ij) -- different day same person
                    ss2 += d1[i1, :y] * d1[i2, :y]
                    ss2 += d1[i1+m, :y] * d1[i2+m, :y]
                    qq2 += 2
                end
            end
        end
    end

    return [ss1, ss2, ss3]
end

# Rotate so that the scores for different factors are orthogonal.
function rotate_orthog(u::AbstractMatrix, v::AbstractMatrix)

    # Handle the one factor case separately.
    if size(u, 2) == 1
        f = norm(u)
        return tuple(u ./ f, v .* f)
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

    return tuple(u, v)
end

function objective(v3, dyad_info)
    vd1 = vardecomp(v3[:, 1], dyad_info, false)
    vd2 = vardecomp(v3[:, 2], dyad_info, false)
    vd3 = vardecomp(v3[:, 3], dyad_info, false)

    ip = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

    va = []
    for ii in ip
        z1 = vd1[ii[1]] - vd1[ii[2]] - vd1[ii[3]]
        z2 = vd2[ii[2]] - vd2[ii[1]] - vd2[ii[3]]
        z3 = vd3[ii[3]] - vd3[ii[1]] - vd3[ii[2]]
        push!(va, z1 + z2 + z3)
    end

    return maximum(va)
end

function xsearch(u3::AbstractMatrix, v3::AbstractMatrix, randomize::Bool)
    mm = objective(center_subjects(v3), dyad_info)
    println("Initial objective=", mm)
    for ii = 1:50
        println("ii=", ii)
        # Generate a random skew-symmetric matrix, to determine
        # the search direction.
        h = randn(3, 3)
        h = h - h'
        ah, bh = eigen(h)

        step = 1.0
        sign = 1.0
        for k = 1:10

            dt = real.(bh * diagm(exp.(sign * step * ah)) * inv(bh))
            u3x = u3 * dt
            v3x = v3 * dt
            mm1 = objective(center_subjects(v3x), dyad_info)

            # Accept the point
            if mm1 >= mm
                println("  ", mm1, " ", k)
                u3, v3 = u3x, v3x
                mm = mm1
                break
            end
            step /= 2
            sign *= -1
        end
    end

    return tuple(u3, v3, mm)
end

# Plot the means
function plot_means(ifig)
    PyPlot.clf()
    ax = PyPlot.axes([0.12, 0.1, 0.67, 0.8])
    PyPlot.grid(true)
    hh = (1:1440) / 60
    PyPlot.plot(hh, pt_mean[:, :hrtrt], "-", label = "Patient")
    PyPlot.plot(hh, cg_mean[:, :hrtrt], "-", label = "Caregiver")
    ha, lb = ax.get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.xlabel("Time (hours from midnight)", size = 15)
    PyPlot.ylabel("Mean heart rate", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1
    return ifig
end

function plot_vpars(u::AbstractMatrix, v::AbstractMatrix, dyad_info, ifig::Int)

    m = 10
    xx = range(-1, 1, length = m)
    yy = range(-1, 0, length = m)

    hmat = [zeros(m, m) for _ = 1:3]
    for (i, x) in enumerate(xx)
        for (j, y) in enumerate(yy)
            vd = vardecomp(x * v[:, 1] + y * v[:, 2], dyad_info, false)
            vd ./= sum(vd)
            hmat[1][i, j] = vd[1]
            hmat[2][i, j] = vd[2]
            hmat[3][i, j] = vd[3]
        end
    end

    for j = 1:3
        PyPlot.clf()
        PyPlot.imshow(hmat[j], interpolation = "nearest")
        PyPlot.colorbar()
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    return ifig
end


# Plot the component loadings
function plot_loadings(u::AbstractMatrix, v::AbstractMatrix, icc, ifig::Int)
    PyPlot.clf()
    PyPlot.figure(figsize = (8, 5))
    ax = PyPlot.axes([0.1, 0.1, 0.65, 0.8])
    PyPlot.grid(true)
    hh = (1:1440) / 60
    PyPlot.plot(
        hh,
        u[:, 1],
        "-",
        label = @sprintf("%.2f/%.2f/%.2f", icc[1][1], icc[1][2], icc[1][3]),
    )
    PyPlot.plot(
        hh,
        u[:, 2],
        "-",
        label = @sprintf("%.2f/%.2f/%.2f", icc[2][1], icc[2][2], icc[2][3]),
    )
    PyPlot.plot(
        hh,
        u[:, 3],
        "-",
        label = @sprintf("%.2f/%.2f/%.2f", icc[3][1], icc[3][2], icc[3][3]),
    )
    ha, lb = ax.get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    leg.set_title("ICC")
    PyPlot.xlabel("Time (hours from midnight)", size = 15)
    PyPlot.ylabel("Loading", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1
    return ifig
end

function plot_scores_groups(v::AbstractMatrix, ifig)

    n = size(v, 1)
    m = div(n, 2)

    xx = []
    for j = 1:3
        push!(xx, v[1:m, j])
        push!(xx, v[m+1:end, j])
    end

    PyPlot.clf()
    PyPlot.boxplot(xx)
    ax = PyPlot.gca()
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["CG-1", "Pt-1", "CG-2", "Pt-2", "CG-3", "Pt-3"])
    for x in ax.get_xticklabels()
        x.set_rotation(-50)
    end
    PyPlot.ylabel("Factor score", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1
end

# Plot the scores
function plot_scores_pairs(ifig)
    for j = 1:3

        x, y = Float64[], Float64[]
        x0, y0 = Float64[], Float64[]
        n = size(xr3_v, 1)
        m = div(n, 2)

        for i = 1:m
            push!(x0, xr3_v[i, j])
            push!(y0, xr3_v[i+m, j])
            ii = rand(1:m)
            jj = rand(1:m-1)
            if jj >= ii
                jj += 1
            end
            push!(x, xr3_v[ii, j])
            push!(y, xr3_v[jj+m, j])
        end

        PyPlot.clf()
        ax = PyPlot.axes([0.1, 0.1, 0.7, 0.8])
        PyPlot.grid(true)

        PyPlot.plot(
            x,
            y,
            "o",
            color = "grey",
            mfc = "none",
            alpha = 0.3,
            label = "Non-dyad",
        )
        PyPlot.plot(x0, y0, "o", color = "red", mfc = "none", alpha = 0.3, label = "Dyad")

        b = cov(x, y) / var(x)
        a = mean(y) - b * mean(x)
        xp = range(extrema(x)...; step = 0.05)
        yp = a .+ b .* xp
        PyPlot.plot(xp, yp, "-", color = "grey", label = "Non-dyad")

        b = cov(x0, y0) / var(x0)
        a = mean(y0) - b * mean(x0)
        xp0 = range(extrema(x0)...; step = 0.05)
        yp0 = a .+ b .* xp0
        PyPlot.plot(xp0, yp0, "-", color = "red", label = "Dyad")

        ha, lb = ax.get_legend_handles_labels()
        leg = PyPlot.figlegend(ha, lb, "center right")
        leg.draw_frame(false)
        PyPlot.xlabel("Caregiver", size = 15)
        PyPlot.ylabel("Patient", size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

    end
    return ifig
end

function plot_loading_range(ifig)

    hh = (1:1440) / 60
    cols = ["blue", "lightblue", "grey", "orange", "red"]

    for k = 1:2
        mn = k == 1 ? cg_mean : pt_mean
        mn = mn[:, :hrtrt]
        for j = 1:3
            PyPlot.clf()
            PyPlot.grid(true)
            jj = 1
            for s in [-2, -1, 0, 1, 2]
                PyPlot.plot(hh, mn + s * xr3_u[:, j], "-", color = cols[jj])
                jj += 1
            end
            PyPlot.xlabel("Time (hours from midnight)", size = 15)
            PyPlot.ylabel("Mean + component $(j)", size = 15)
            PyPlot.title(k == 1 ? "Caregiver" : "Patient")
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    return ifig
end

cg_mean = open(GzipDecompressorStream, "cg_mean_$(src).csv.gz") do io
    CSV.read(io, DataFrame)
end

pt_mean = open(GzipDecompressorStream, "pt_mean_$(src).csv.gz") do io
    CSV.File(io) |> DataFrame
end

# 1 component solution
r1_u = open(GzipDecompressorStream, "$(src)_1_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end
r1_v = open(GzipDecompressorStream, "$(src)_1_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

# 2 component solution
r2_u = open(GzipDecompressorStream, "$(src)_2_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end
r2_v = open(GzipDecompressorStream, "$(src)_2_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

# 3 component solution
r3_u = open(GzipDecompressorStream, "$(src)_3_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end
r3_v = open(GzipDecompressorStream, "$(src)_3_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r1_u, r1_v = rotate_orthog(r1_u, r1_v)
r2_u, r2_v = rotate_orthog(r2_u, r2_v)
r3_u, r3_v = rotate_orthog(r3_u, r3_v)

pve1 = 1 - sum(abs2, skipmissing(r1_u * r1_v' - mx)) / sum(abs2, skipmissing(mx))
pve2 = 1 - sum(abs2, skipmissing(r2_u * r2_v' - mx)) / sum(abs2, skipmissing(mx))
pve3 = 1 - sum(abs2, skipmissing(r3_u * r3_v' - mx)) / sum(abs2, skipmissing(mx))

#randomize = false
#r3_u, r3_v, mm = xsearch(r3_u, r3_v, randomize)
icc = [vardecomp(r3_v[:, j], dyad_info, false) for j = 1:3]

ifig = 0
ifig = plot_means(ifig)
ifig = plot_loadings(r3_u, r3_v, icc, ifig)
ifig = plot_vpars(r2_u, r2_v, dyad_info, ifig)
#ifig = plot_scores_groups(r3_v, ifig)
#ifig = plot_scores_pairs(ifig)
#ifig = plot_loading_range(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=factor_$(src).pdf $f`
run(c)

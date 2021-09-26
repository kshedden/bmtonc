using GZip, CSV, Tables, LinearAlgebra, Statistics, PyPlot, Printf

rm("plots", recursive = true, force = true)
mkdir("plots")

include("factor_setup.jl")

src = "onc"
mx, dyad_info = factor_setup(src)

# Center the first half of the rows of u (caregivers) and the second
# half of the rows of u (patients).
function center(u)
    u = copy(u)
    n = size(u, 1)
    m = div(n, 2)

    for j = 1:size(u, 2)
        u[1:m, j] = u[1:m, j] .- mean(u[1:m, j])
        u[m+1:end, j] = u[m+1:end, j] .- mean(u[m+1:end, j])
    end

    return u
end

function vardecomp(v, dyad_info, randomize)

    # i=dyad, j=person, k=day
    # y_ijk = a_i + b_ij + c_ik + e_ijk 

    y = v .- mean(v)
    y = y ./ std(y)

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

        # Var(a_i)
        for i1 = 1:m
            for i2 = 1:m
                if i1 != i2
                    ss1 += d1[i1, :y] * d1[m+i2, :y]
                    qq1 += 1
                end
            end
        end

        # Var(b_ij)
        for i1 = 1:m
            for i2 = 1:m
                if i1 != i2
                    ss2 += d1[i1, :y] * d1[i2, :y]
                    ss2 += d1[i1+m, :y] * d1[i2+m, :y]
                    qq2 += 2
                end
            end
        end

        # Var(c_ik)
        for i1 = 1:m
            ss3 += d1[i1, :y] * d1[i1+m, :y]
            qq3 += 1
        end
    end

    ss1 /= qq1
    ss2 /= qq2
    ss3 /= qq3

    return tuple(ss1, ss2, ss3)

end

# Rotate so that the scores for different factors are orthogonal.
# u1*s1*v1 = u
function rotate_orthog(u, v)

    if size(u, 2) == 1
        f = norm(u)
        return tuple(u ./ f, v .* f)
    end

    u1, s1, v1 = svd(v)
    u = u * v1 * diagm(s1)
    v = u1

    n = size(v, 1)
    u ./= sqrt(n)
    v .*= sqrt(n)
    return tuple(u, v)
end

function rotate_random(u, v)
    p = size(v, 2)
    r = randn(100, p)
    r = r' * r
    r, _, _ = svd(r)
    return tuple(u * r, v * r)
end


function xsearch(u3, v3, randomize)
    mm = nothing
    for ii = 1:100
        u3x, v3x = rotate_random(u3, v3)
        v3xc = center(v3x)
        vd1 = vardecomp(v3xc[:, 1], dyad_info, randomize)
        vd2 = vardecomp(v3xc[:, 2], dyad_info, randomize)
        vd3 = vardecomp(v3xc[:, 3], dyad_info, randomize)
        mm1 = sort([vd1[1] + vd1[3], vd2[1] + vd2[3], vd3[1] + vd3[3]])
        mm1 = mm1[2] + 2 * mm1[3]
        if ii == 1 || mm1 >= mm
            u3, v3 = u3x, v3x
            println(mm1)
            mm = mm1
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


# Plot the component loadings
function plot_loadings(ifig)
    PyPlot.clf()
    PyPlot.figure(figsize = (8, 5))
    ax = PyPlot.axes([0.1, 0.1, 0.65, 0.8])
    PyPlot.grid(true)
    hh = (1:1440) / 60
    PyPlot.plot(
        hh,
        xr3_u[:, 1],
        "-",
        label = @sprintf("%.2f/%.2f/%.2f", icc[1][1], icc[1][2], icc[1][3]),
    )
    PyPlot.plot(
        hh,
        xr3_u[:, 2],
        "-",
        label = @sprintf("%.2f/%.2f/%.2f", icc[2][1], icc[2][2], icc[2][3]),
    )
    PyPlot.plot(
        hh,
        xr3_u[:, 3],
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

function plot_scores_groups(ifig)

    n = size(xr3_v, 1)
    m = div(n, 2)

    xx = []
    for j = 1:3
        push!(xx, xr3_v[1:m, j])
        push!(xx, xr3_v[m+1:end, j])
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

cg_mean = GZip.open("cg_mean_$(src).csv.gz") do io
    CSV.File(io) |> DataFrame
end

pt_mean = GZip.open("pt_mean_$(src).csv.gz") do io
    CSV.File(io) |> DataFrame
end

r1_u = GZip.open("$(src)_1_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r1_v = GZip.open("$(src)_1_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r2_u = GZip.open("$(src)_2_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r2_v = GZip.open("$(src)_2_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r3_u = GZip.open("$(src)_3_u.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r3_v = GZip.open("$(src)_3_v.csv.gz") do io
    CSV.File(io) |> Tables.matrix
end

r1_u, r1_v = rotate_orthog(r1_u, r1_v)
r2_u, r2_v = rotate_orthog(r2_u, r2_v)
r3_u, r3_v = rotate_orthog(r3_u, r3_v)

pve1 = 1 - sum(x -> x^2, skipmissing(r1_u * r1_v' - mx)) / sum(x -> x^2, skipmissing(mx))
pve2 = 1 - sum(x -> x^2, skipmissing(r2_u * r2_v' - mx)) / sum(x -> x^2, skipmissing(mx))
pve3 = 1 - sum(x -> x^2, skipmissing(r3_u * r3_v' - mx)) / sum(x -> x^2, skipmissing(mx))

randomize = false
xr3_u, xr3_v, mm = xsearch(r3_u, r3_v, randomize)

icc = [vardecomp(xr3_v[:, j], dyad_info, false) for j = 1:3]

ifig = 0
ifig = plot_means(ifig)
ifig = plot_loadings(ifig)
ifig = plot_scores_groups(ifig)
ifig = plot_scores_pairs(ifig)
ifig = plot_loading_range(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=factor_$(src).pdf $f`
run(c)

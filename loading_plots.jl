using PyPlot, SupportPoints, CSV, CodecZlib, DataFrames, Printf
using Tables, NearestNeighbors, Distributions, LinearAlgebra
using Statistics

include("factor_utils.jl")
include("factor_setup.jl")

rm("plots", recursive = true, force = true)
mkdir("plots")

# Number of factors in the low-rank representation
qq = 5

# Time of day, in hours
hh = collect(range(1, 1440) / 60)

function load_data(src, qq)

    # Caregiver mean
    f = @sprintf("cg_mean_%s.csv.gz", src)
    cg_mean = open(GzipDecompressorStream, f) do io
        CSV.read(io, DataFrame)
    end

    # Patient mean
    f = @sprintf("pt_mean_%s.csv.gz", src)
    pt_mean = open(GzipDecompressorStream, f) do io
        CSV.read(io, DataFrame)
    end

    # All loadings
    f = @sprintf("results/%s_%d_loadall.csv.gz", src, qq)
    loadall = open(GzipDecompressorStream, f) do io
        CSV.read(io, Tables.matrix)
    end

    # Observed data ICC values for every loading vector in loadall
    f = @sprintf("results/%s_%d_vda.csv.gz", src, qq)
    vda = open(GzipDecompressorStream, f) do io
        CSV.read(io, Tables.matrix)
    end

    # Observed data variance values for every loading vector in loadall
    f = @sprintf("results/%s_%d_vdx.csv.gz", src, qq)
    vdx = open(GzipDecompressorStream, f) do io
        CSV.read(io, Tables.matrix)
    end

    # Observed data ICC values for every loading vector in loadall
    vdar = []
    for k = 1:100
        f = @sprintf("results/%s_%d_vda_%d.csv.gz", src, qq, k)
        va = open(GzipDecompressorStream, f) do io
            CSV.read(io, Tables.matrix)
        end
        push!(vdar, va)
    end

    return cg_mean, pt_mean, loadall, vda, vdx, vdar
end

function plot_means(src, cg_mean, pt_mean, ifig)
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.68, 0.8])
    PyPlot.grid(true)
    PyPlot.plot(hh, cg_mean[:, :hrtrt], "-", label = "Caregiver", color = "orange")
    PyPlot.plot(hh, pt_mean[:, :hrtrt], "-", label = "Patient", color = "purple")
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

function plot_components(src, loadall, vda, vdx, cg_mean, pt_mean, ifig)
    hh = collect(range(1, 1440) / 60)
    pp = collect(range(0.1, 0.9, length = 9))
    for j = 1:4

        _, ii = findmax(vda[j, :])

        PyPlot.clf()
        PyPlot.grid(true)
        for (i, s) in enumerate([-2, -1, 0, 1, 2])
            col = PyPlot.cm.plasma(i / 6)
            f = s * sqrt(vdx[j, ii])
            PyPlot.plot(hh, cg_mean[:, :hrtrt] + f * loadall[:, ii], "-", color = col)
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
        for (i, s) in enumerate([-2, -1, 0, 1, 2])
            col = PyPlot.cm.plasma(i / 6)
            f = s * sqrt(vdx[j, ii])
            PyPlot.plot(hh, pt_mean[:, :hrtrt] + f * loadall[:, ii], "-", color = col)
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

function pvalues(vda, vdar, j)

    mx = maximum(vda[j, :])
    mxr = [maximum(r[j, :]) for r in vdar]
    p1 = mean(mxr .>= mx)

    mm = zeros(length(vdar), size(vda, 2))
    for i = 1:length(vdar)
        mm[i, :] = vdar[i][j, :] .>= vda[j, :]
    end
    p2 = mean(mm, dims = 1)[:]

    return p1, p2
end

function factor_summary(vda, vdar, src, qq, out, ifig)

    mx, dyad_info = factor_setup(src)
    write(out, @sprintf("%s %d total person/days\n", uppercase(src), size(mx, 1)))

    for j = 1:3

        p1, p2 = pvalues(vda, vdar, j)

        write(out, @sprintf("%s component %d:\n", uppercase(src), j))
        mx = maximum(vda[j, :])
        write(out, @sprintf("Maximum ICC over support points: %.3f\n", mx))
        write(
            out,
            @sprintf(
                "Minimum ICC over significant support points: %.3f\n",
                minimum(vda[j, p2.<=0.05])
            )
        )
        mxr = [maximum(r[j, :]) for r in vdar]
        p = mean(mxr .>= mx)
        write(
            out,
            @sprintf(
                "  %.2f fraction of randomized data have maximal ICC exceeding the maximal ICC in observed data.\n",
                p
            )
        )
        mm = zeros(length(vdar), size(vda, 2))
        for i = 1:length(vdar)
            mm[i, :] = vdar[i][j, :] .>= vda[j, :]
        end
        p = mean(mm, dims = 1)
        write(
            out,
            @sprintf(
                "  %.0f number of support points with ICC significantly greater than zero.\n",
                sum(p .< 0.05)
            )
        )
        write(out, "\n")
    end

    return ifig
end

function spaghetti(loadall, vda, vdar, src, qq, ifig)

    # Time of day in hours
    hh = collect(range(1, 1440) / 60)

    # Spaghetti plot for each component
    for k = 1:4

        p1, p2 = pvalues(vda, vdar, k)

        PyPlot.clf()
        PyPlot.grid(true)
        for i = 1:size(loadall, 2)
            col = p2[i] < 0.05 ? "black" : "grey"
            sym = p2[i] < 0.05 ? "-" : "--"
            PyPlot.plot(hh, loadall[:, i], sym, alpha = 0.8, color = col)
        end
        PyPlot.title(["Dyad", "Person", "Dyad/day", "Residual"][k])
        PyPlot.xlabel("Time of day", size = 15)
        PyPlot.ylabel("Loading", size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end

    return ifig
end


function flip(z)

    n = size(z, 2)

    needsflip = function (j)
        mn = mean(z, dims = 2)[:]
        return sum(abs2, z[:, j] - mn) > sum(abs2, -z[:, j] - mn)
    end

    for k = 1:3
        for j = 1:n
            if needsflip(j)
                z[:, j] .*= -1
            end
        end
    end

    return z
end

function main(ifig)

    out = open("factor_map_$(qq).txt", "w")

    for src in ["bmt", "onc"]
        cg_mean, pt_mean, loadall, vda, vdx, vdar = load_data(src, qq)
        ifig = plot_means(src, cg_mean, pt_mean, ifig)
        ifig = plot_components(src, loadall, vda, vdx, cg_mean, pt_mean, ifig)
        ifig = spaghetti(loadall, vda, vdar, src, qq, ifig)
        ifig = factor_summary(vda, vdar, src, qq, out, ifig)
    end

    close(out)

    return ifig
end

ifig = 0
ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=loading_plots_$(qq).pdf $f`
run(c)

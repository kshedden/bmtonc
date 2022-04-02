using CodecZlib, CSV, Tables, LinearAlgebra, Statistics
using Printf, Distributions, Random

Random.seed!(42)

rm("plots", recursive = true, force = true)
mkdir("plots")

include("factor_utils.jl")
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

function xfill(uu::AbstractMatrix, vv::AbstractMatrix, np::Int, dyad_info)

    n, q = size(uu)
    @assert size(uu, 2) == size(vv, 2)

    loadall = zeros(1440, np)
    vdb = Float64[0, 0, 0, 0]
    vdc = zeros(n, 4)
    vds = zeros(size(vv, 1), 4)
    vdm = zeros(4, np)
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
        vdm[:, ii] = vd
        loadall[:, ii] = uu[:, 1]

        for j = 1:4
            if ii == 1 || vd[j] > vdb[j]
                vdb[j] = vd[j]
                vdc[:, j] = uu[:, 1]
                vds[:, j] = vvx
            end
        end
    end

    return vdm, vdb, vdc, vds, loadall
end

function save_results(loadall::AbstractMatrix, vdm::AbstractMatrix, 
                      vdb::AbstractVector, vds::AbstractMatrix, 
                      vdc::AbstractMatrix, src::String, qq::Int)

	f = @sprintf("%s_%d_loadall.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(loadall))
	end

	f = @sprintf("%s_%d_vdm.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(vdm))
	end

	f = @sprintf("%s_%d_vdb.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(vdb))
	end

	f = @sprintf("%s_%d_vds.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(vds))
	end

	f = @sprintf("%s_%d_vdc.csv.gz", src, qq)
	open(GzipCompressorStream, f, "w") do io
		CSV.write(io, Tables.table(vdc))
	end
end

function load_start(src, qq)
    # Loadings
    uu = open(GzipDecompressorStream, "$(src)_$(qq)_u.csv.gz") do io
        CSV.File(io) |> Tables.matrix
    end

    # Scores
    vv = open(GzipDecompressorStream, "$(src)_$(qq)_v.csv.gz") do io
        CSV.File(io) |> Tables.matrix
    end

    uu, vv, mn = rotate_orthog(uu, vv)
	return uu, vv, mn
end

# Use the solutions with qq factors
qq = 5

# Number of points to generate
npt = 2000

function main()
    for src in ["bmt", "onc"]
	    mx, dyad_info = factor_setup(src)
    	uu, vv, mn = load_start(src, qq)
	    vdm, vdb, vdc, vds, loadall = xfill(uu, vv, npt, dyad_info)
		save_results(loadall, vdm, vdb, vds, vdc, src, qq)
    end
end

main()

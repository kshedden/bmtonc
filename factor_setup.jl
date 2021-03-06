using DataFrames, CodecZlib, CSV, Dates, Statistics, Tables

pa = "/home/kshedden/data/Sung_Choi"
pal = joinpath(pa, "long")
paw = joinpath(pa, "wide")

# Return a single array containing caregivers then patients,
# stacked vertically.  There are 1440 columns corresponding
# to minutes within the day.  Since these data are used for
# factor analysis, the rows are mean centered.
function factor_setup(src::String; save::Bool = false)

    @assert src == "onc" || src == "bmt"

    df = open(GzipDecompressorStream, "$(pal)/$(src).csv.gz") do io
        CSV.read(io, DataFrame)
    end

    # Select at most one value per minute for each person/day
    df[:, :Timex] = round.(df[:, :Time], Dates.Minute)
    df = combine(first, groupby(df, [:id_caregiver, :Timex]))

    df[:, :Date] = Date.(df[:, :Time])

    cg, pt, dyad_info = make_data(df)

    if save
        # Save the caregiver means
        cg_mean = dmean(cg)
        open(GzipCompressorStream, "cg_mean_$(src).csv.gz", "w") do io
            CSV.write(io, cg_mean)
        end

        # Save the patient means
        pt_mean = dmean(pt)
        open(GzipCompressorStream, "pt_mean_$(src).csv.gz", "w") do io
            CSV.write(io, pt_mean)
        end

        # Save the data
        for (k, ds) in enumerate([cg, pt])
            dz = DataFrame(copy(ds'), :auto)
            println(size(dz))
            rename!(dz, ["Minute$(j)" for j = 1:1440])
            dz[:, :ID] = dyad_info.ID
            dz[:, :Day] = dyad_info.Day
            c = vcat(["ID", "Day"], ["Minute$(j)" for j = 1:1440])
            dz = dz[:, c]
            xp = ["cg", "pt"][k]
            open(GzipCompressorStream, "$(paw)/$(src)_$(xp)_wide.csv.gz", "w") do io
                CSV.write(io, dz)
            end
        end
    end

    # Center the caregivers and patients for each minute, and stack them vertically.
    cgc = rowcenter(cg)
    ptc = rowcenter(pt)
    mx = hcat(cgc, ptc)

    return tuple(mx, dyad_info)
end

function getrow(minut, hrtrt)
    rr = Array{Union{Float64,Missing},1}(undef, 1440)
    for i in eachindex(minut)
        if !ismissing(hrtrt[i])
            rr[1+minut[i]] = hrtrt[i]
        end
    end
    return rr
end

# Create aligned arrays for the patients and caregivers, where the
# rows are dyads and the columns are minutes within a day.
function make_data(df)

    pt_rows = []
    cg_rows = []
    dyad_id = Int[]
    dyad_day = Int[]

    ii = 0
    for ds in groupby(df, :id_caregiver)
        ii += 1

        jj = 0
        for dat in groupby(ds, :Date)
            jj += 1

            push!(dyad_id, ii)
            push!(dyad_day, jj)

            # The minute within a day, from 0 to 1439.
            minut = 60 * hour.(dat[:, :Time]) + minute.(dat[:, :Time])

            rr = getrow(minut, dat[:, :HR_caregiver])
            push!(cg_rows, rr)

            rr = getrow(minut, dat[:, :HR_patient])
            push!(pt_rows, rr)
        end
    end

    cg = hcat(cg_rows...)
    pt = hcat(pt_rows...)
    dyad_info = DataFrame(:ID => dyad_id, :Day => dyad_day)

    return tuple(cg, pt, dyad_info)
end

# Return the mean of observed values for each minute within the day.
function dmean(dx)
    mn = [mean(skipmissing(v)) for v in eachrow(dx)]
    mn = DataFrame(:minute => range(1, 1440, length = 1440), :hrtrt => mn)
    return mn
end

function rowcenter(x)
    for i = 1:size(x, 1)
        x[i, :] = x[i, :] .- mean(skipmissing(x[i, :]))
    end
    return x
end

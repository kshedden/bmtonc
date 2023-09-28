using DataFrames
using CodecZlib
using CSV
using Dates
using Statistics
using Tables
using Printf

import Statistics.median
median(dateTimeArray::Array{DateTime,1}) =
    DateTime(Dates.UTM(Int(round(median(Dates.value.(dateTimeArray))))))

# Aggregate to this number of minutes
tres = 5
@assert 60 % tres == 0

# Number of distinct time units
ntime = div(24*60, tres)

pa = "/home/kshedden/data/Sung_Choi"
pal = joinpath(pa, "long")
paw = joinpath(pa, "wide")

# Return a single array containing caregivers then patients,
# stacked vertically.  There are 'ntime' columns corresponding
# to minutes within the day.  Since these data are used for
# factor analysis, the rows are mean centered.
function make_wide(src::String)

    @assert src == "onc" || src == "bmt"

    # Load the long-form data
    df = open(GzipDecompressorStream, "$(pal)/$(src).csv.gz") do io
        CSV.read(io, DataFrame)
    end

    # Select at most one value per minute for each person/day
    df[:, :Timex] = round.(df[:, :Time], Dates.Minute(tres))
    dx = combine(groupby(df, [:id_caregiver, :id_patient, :Timex]), :HR_caregiver=>median,
                 :HR_patient=>median)
    dx = rename(dx, :HR_caregiver_median=>:HR_caregiver, :HR_patient_median=>:HR_patient)
    dx[:, :Date] = Date.(dx[:, :Timex])
    cg, pt, id_patient, id_caregiver, dates = make_aligned_data(dx)

    # Time value labels
    q = div(60, tres)
    hr = repeat(0:23, inner=q)
    mn = repeat(0:tres:60, outer=24)
    tiv = [@sprintf("%02d:%02d", h, m) for (h,m) in zip(hr, mn)]

    # Save the data
    idx = [id_caregiver, id_patient]
    for (k, ds) in enumerate([cg, pt])
        dz = DataFrame(copy(ds'), tiv)
        dz[:, :ID] = idx[k]
        dz[:, :Date] = dates

        # Reorder the columns
        c = vcat(["ID", "Date"], tiv)
        dz = dz[:, c]

        xp = ["cg", "pt"][k]
        open(GzipCompressorStream, "$(paw)/$(src)_$(xp)_wide.csv.gz", "w") do io
            CSV.write(io, dz)
        end
    end
end

function make_aligned_data(df)

    pt, cg, dates = [], [], []
    id_patient, id_caregiver = [], []
    for ds in groupby(df, [:id_caregiver, :Date])

        @assert length(unique(ds[:, :id_patient])) == 1

        push!(id_patient, first(ds[:, :id_patient]))
        push!(id_caregiver, first(ds[:, :id_caregiver]))
        push!(dates, first(ds[:, :Date]))

        # A full day to merge onto.
        tix = 1:div(1440, tres)
        da = DataFrame(tix=tix)

        h = [x.value for x in Hour.(ds[:, :Timex])]
        m = [x.value for x in Minute.(ds[:, :Timex])]
        tix = div(60, tres) * h + div.(m, tres) .+ 1
        ds[:, :tix] = tix

        da = leftjoin(da, ds, on=:tix)
        da = sort(da, :tix)
        push!(pt, da[:, :HR_patient])
        push!(cg, da[:, :HR_caregiver])
    end

    cg = hcat(cg...)
    pt = hcat(pt...)

    return cg, pt, id_patient, id_caregiver, dates
end

make_wide("bmt")
make_wide("onc")

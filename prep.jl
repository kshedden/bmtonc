using CSV, GZip, DataFrames, Dates

# The design information files and file long-form data are here
pa = "/home/kshedden/data/Sung_Choi"

# The raw data files are here
px = "/nfs/turbo/umms-sungchoi/ROADMAP"

# Read the oncology study design information
onc_info = GZip.open(joinpath(pa, "roadmap_onc.csv.gz")) do io
    CSV.read(io, DataFrame)
end

# Read the roadmap study design information
bmt_info = GZip.open(joinpath(pa, "roadmap_bmt.csv.gz")) do io
    CSV.read(io, DataFrame)
end

# Make the subject id's consistent with the form ID-#.
function clean_header(dx)
    for x in names(dx)
        s1 = string(x)
        s2 = replace(s1, "ID- " => "ID-")
        rename!(dx, s1 => s2)
    end
end

clean_header(onc_info)
clean_header(bmt_info)

function read_all(pa, id1, id2)

    isdir(pa) || return nothing
    fi = readdir(pa)

    dtf = DateFormat("y-m-d H:M:S")

    dl = []
    for f in fi
        fx = readdir(joinpath(pa, f))

        # If people have multiple files, they appear to be identical,
        # so just choose one of them
        fy = joinpath(pa, f, fx[1])

        dd = open(fy) do io
            CSV.read(io, DataFrame, header = false)
        end
        dd[!, 1] = [DateTime(x, dtf) for x in dd[:, 1]]
        rename!(dd, [:Time, :HR, :DBId])
        dd = select(dd, Not(:DBId))
        push!(dl, dd)
    end

    dd = vcat(dl...)
    dd[!, :id] .= id2
    dd = sort(dd, :Time)
    return dd

end

function make_long(info, sname)

    otn = joinpath(pa, "long", "$(sname).csv.gz")
    out = GZip.open(otn, "w")
    app = false

    # Process each patient/caregiver dyad
    for r in eachrow(info)

        # Caregiver
        r1s, r2s = string(r[1]), strip(r[2])
        p1 = joinpath(px, r1s, r2s)
        d1 = read_all(p1, r1s, r2s)
        if isnothing(d1)
            println("skipping $(r1s) $(r2s)")
            continue
        end
        d1 = rename(d1, :HR => :HR_caregiver, :id => :id_caregiver)

        # Patient
        r3s, r4s = string(r[3]), strip(r[4])
        p2 = joinpath(px, r3s, r4s)
        d2 = read_all(p2, r3s, r4s)
        if isnothing(d2)
            println("skipping $(r3s) $(r4s)")
            continue
        end
        d2 = rename(d2, :HR => :HR_patient, :id => :id_patient)

        dd = outerjoin(d1, d2, on = :Time)
        dd = sort(dd, :Time)

        # After merging there will be some missing values in these columns
        dd[!, :id_caregiver] .= r2s
        dd[!, :id_patient] .= r4s

        CSV.write(out, dd, append = app)
        app = true

    end

    close(out)

end

make_long(onc_info, "onc")
make_long(bmt_info, "bmt")

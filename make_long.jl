using CSV
using CodecZlib
using DataFrames
using Dates
using Printf

# The design information files and file long-form data are here
pa = "/home/kshedden/data/Sung_Choi"

# The raw data files are here
pax = "/nfs/turbo/umms-sungchoi/ROADMAP_ALL"

# Map subject id's to the directory containing their data.
function get_idmap()
    idmap = Dict{String,String}()
    dd = readdir(pax)
    for d in dd
        dp = joinpath(pax, d)
        if isdir(dp)
            f = readdir(dp)
            for g in f
                if isdir(joinpath(pax, d, g))
                    idmap[g] = d
                end
            end
        end
    end
    return idmap
end

idmap = get_idmap()

# Read the oncology study design information
onc_info = open(GzipDecompressorStream, joinpath(pa, "roadmap_onc.csv.gz")) do io
    CSV.read(io, DataFrame)
end

# Read the roadmap study design information
bmt_info = open(GzipDecompressorStream, joinpath(pa, "roadmap_bmt.csv.gz")) do io
    CSV.read(io, DataFrame)
end

function clean_info(info)
    info = info[:, ["pt_rm_access_code", "cg_rm_access_code"]]
    info = rename(info, "pt_rm_access_code"=>"pt_id", "cg_rm_access_code"=>"cg_id")
    return info
end

onc_info = clean_info(onc_info)
bmt_info = clean_info(bmt_info)


function read_all_heart(id)

    if !haskey(idmap, id)
        return nothing
    end
    pa = joinpath(pax, idmap[id], id)
    fi = readdir(pa)

    dtf = DateFormat("y-m-d H:M:S")
    println("Reading $(pa)")
    dl = []
    for f in fi
        paf = joinpath(pa, f)
        if !isdir(paf)
            continue
        end
        fx = readdir(paf)
        fx = [a for a in fx if occursin("HEART", a)]
        if length(fx) == 0
            continue
        end

        # If people have multiple heartrate files choose the largest of them
        fs = [filesize(pa, f, x) for x in fx]
        ii = argmax(fs)
        fy = joinpath(pa, f, fx[ii])

        dd = open(fy) do io
            CSV.read(io, DataFrame, header = false)
        end
        dd[!, 1] = [DateTime(x, dtf) for x in dd[:, 1]]
        rename!(dd, [:Time, :Time2, :HR, :DBId, :Unknown])
        dd = select(dd, [:Time, :HR])
        push!(dl, dd)
    end

    if length(dl) == 0
        return nothing
    end
    dd = vcat(dl...)
    dd[:, :id] .= id
    dd = sort(dd, :Time)
    return dd
end

function read_all_mood(id)

    if !haskey(idmap, id)
        return nothing
    end
    pa = joinpath(pax, idmap[id], id)
    fi = readdir(pa)

    dtf = DateFormat("y-m-d H:M:S")
    println("Reading $(pa)")
    dl = []
    for f in fi
        paf = joinpath(pa, f)
        if !isdir(paf)
            continue
        end
        fx = readdir(paf)
        fx = [a for a in fx if occursin("MOOD", a)]
        if length(fx) == 0
            continue
        end

        # If people have multiple mood files choose the largest of them
        fs = [filesize(pa, f, x) for x in fx]
        ii = argmax(fs)
        fy = joinpath(pa, f, fx[ii])

        dd = open(fy) do io
            CSV.read(io, DataFrame, header = false)
        end
        dd[!, 1] = [DateTime(x, dtf) for x in dd[:, 1]]
        dd[!, 2] = [DateTime(x, dtf) for x in dd[:, 2]]
        rename!(dd, [:Time, :Time2, :mood1, :mood2, :mood3])
        dd = select(dd, [:Time, :mood1, :mood2, :mood3])
        push!(dl, dd)
    end

    if length(dl) == 0
        return nothing
    end
    dd = vcat(dl...)
    dd[:, :id] .= id
    dd[!, :Time] = [Date(x) for x in dd[:, :Time]]
    dd = sort(dd, :Time)
    return dd
end

function make_long_mood(info, sname)

    otn = joinpath(pa, "long", "$(sname)_mood.csv.gz")
    out = GzipCompressorStream(open(otn, "w"))
    otn = joinpath("$(sname)_mood.log")
    outlog = open(otn, "w")
    app = false

    write(outlog, "$(sname) cohort\n")
    write(outlog, @sprintf("%d dyads\n", size(info, 1)))

    # Process each patient/caregiver dyad
    for r in eachrow(info)

        # Patient
        pt_id = r[:pt_id]
        d1 = read_all_mood(pt_id)
        if isnothing(d1)
            write(outlog, "skipping patient $(pt_id)\n")
            continue
        end
        d1 = rename(d1, :mood1=>:mood1_patient, :mood2=>:mood2_patient,
                    :mood3=>:mood3_patient, :id=>:id_patient)

        # Caregiver
        cg_id = r[:cg_id]
        d2 = read_all_mood(cg_id)
        if isnothing(d2)
            write(outlog, "skipping caregiver $(cg_id)\n")
            continue
        end
        d2 = rename(d2, :mood1=>:mood1_caregiver, :mood2=>:mood2_caregiver,
                    :mood3=>:mood3_caregiver, :id=>:id_caregiver)

        dd = outerjoin(d1, d2, on=:Time)
        dd = sort(dd, :Time)
        println(first(dd, 10))

        # After merging there will be some missing values in these columns
        dd[!, :id_patient] .= pt_id
        dd[!, :id_caregiver] .= cg_id

        CSV.write(out, dd, append=app)
        write(outlog, @sprintf("%d records for dyad %s %s\n", size(dd, 1),
              pt_id, cg_id))
        app = true
        flush(outlog)
    end

    close(out)
    close(outlog)
end

function make_long_heart(info, sname)

    otn = joinpath(pa, "long", "$(sname).csv.gz")
    out = GzipCompressorStream(open(otn, "w"))
    otn = joinpath("$(sname)_heart.log")
    outlog = open(otn, "w")
    app = false

    write(outlog, "$(sname) cohort\n")
    write(outlog, @sprintf("%d dyads\n", size(info, 1)))

    # Process each patient/caregiver dyad
    for r in eachrow(info)

        # Patient
        pt_id = r[:pt_id]
        d1 = read_all_heart(pt_id)
        if isnothing(d1)
            write(outlog, "skipping patient $(pt_id)\n")
            continue
        end
        d1 = rename(d1, :HR=>:HR_patient, :id=>:id_patient)

        # Caregiver
        cg_id = r[:cg_id]
        d2 = read_all(cg_id)
        if isnothing(d2)
            write(outlog, "skipping caregiver $(cg_id)\n")
            continue
        end
        d2 = rename(d2, :HR=>:HR_caregiver, :id=>:id_caregiver)

        dd = outerjoin(d1, d2, on=:Time)
        dd = sort(dd, :Time)

        # After merging there will be some missing values in these columns
        dd[!, :id_patient] .= pt_id
        dd[!, :id_caregiver] .= cg_id

        CSV.write(out, dd, append=app)
        write(outlog, @sprintf("%d records for dyad %s %s\n", size(dd, 1),
              pt_id, cg_id))
        app = true
        flush(outlog)
    end

    close(out)
    close(outlog)
end

#make_long_heart(onc_info, "onc")
#make_long_heart(bmt_info, "bmt")

make_long_mood(onc_info, "onc")
#make_long_mood(bmt_info, "bmt")

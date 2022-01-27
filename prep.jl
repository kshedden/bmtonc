using CSV, GZip, DataFrames, Dates

# The design information files and file long-form data are here
pa = "/home/kshedden/data/Sung_Choi"

# The raw data files are here
px = "/nfs/turbo/umms-sungchoi/ROADMAP_ALL"

# Read the oncology study design information
onc_info = GZip.open(joinpath(pa, "roadmap_onc.csv.gz")) do io
    CSV.read(io, DataFrame)
end

# Read the roadmap study design information
bmt_info = GZip.open(joinpath(pa, "roadmap_bmt.csv.gz")) do io
    CSV.read(io, DataFrame)
end

function clean_info(info)
	info = info[:, [:caregiver_id, :cg_study_id, :patient_id, :pat_study_id]]
	return info
end

onc_info = clean_info(onc_info)
bmt_info = clean_info(bmt_info)

function read_all(pa, id1, id2)

    isdir(pa) || return nothing
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

        # If people have multiple heartrate files choose one of them
        fy = joinpath(pa, f, fx[1])

        dd = open(fy) do io
            CSV.read(io, DataFrame, header = false)
        end
        dd[!, 1] = [DateTime(x, dtf) for x in dd[:, 1]]
        rename!(dd, [:Time, :Time2, :HR, :DBId, :Unkown])
        dd = select(dd, [:Time, :HR])
        push!(dl, dd)
    end

	if length(dl) == 0
		return nothing
	end
    dd = vcat(dl...)
    dd[:, :id] .= id2
    dd = sort(dd, :Time)
    return dd
end

function make_long(info, sname)

    otn = joinpath(pa, "long", "$(sname)_long.csv.gz")
    out = GZip.open(otn, "w")
    app = false

    # Process each patient/caregiver dyad
    for r in eachrow(info)

        # Caregiver
        cgi, cgf = strip(r[:caregiver_id]), r[:cg_study_id]
        p1 = joinpath(px, string(cgf), cgi)
        d1 = read_all(p1, cgf, cgi)
        if isnothing(d1)
            println("skipping caregiver $(cgf)/$(cgi)")
            continue
        end
        d1 = rename(d1, :HR => :HR_caregiver, :id => :id_caregiver)

        # Patient
        pti, ptf = strip(r[:patient_id]), r[:pat_study_id]
        p2 = joinpath(px, string(ptf), pti)
        d2 = read_all(p2, ptf, pti)
        if isnothing(d2)
            println("skipping patient $(ptf)/$(pti)")
            continue
        end
        d2 = rename(d2, :HR => :HR_patient, :id => :id_patient)

        dd = outerjoin(d1, d2, on = :Time)
        dd = sort(dd, :Time)

        # After merging there will be some missing values in these columns
        dd[!, :id_caregiver] .= cgi
        dd[!, :id_patient] .= pti

        CSV.write(out, dd, append = app)
        app = true
    end

    close(out)
end

make_long(onc_info, "onc")
make_long(bmt_info, "bmt")

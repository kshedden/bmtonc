using Printf
using CSV
using CodecZlib
using DataFrames

# The design information files and file long-form data are here
pa = "/home/kshedden/data/Sung_Choi"

# The raw data files are here
pax = "/nfs/turbo/umms-sungchoi/ROADMAP_ALL"

# Person-level variables that are prefixed with "pt" or "cg".
svars = Dict("onc"=>["gender", "race", "ethnicity", "age_enroll"],
             "bmt"=>["gender", "race", "ethnicity", "age_enroll"])

xvars = Dict("onc"=>["cg_relationship", "cohort"],
             "bmt"=>["cg_relationship", "cg_arm", "cohort"])

function recode_relationship(v)
    v = [strip(x) for x in v]
    v = [lowercase(x) for x in v]
    v = replace.(v, "mom"=>"mother")
    v = replace.(v, "fiance"=>"fiancé")

    rr = Dict("spouse"=>"partner",
              "fiancé"=>"partner",
              "significant other"=>"partner",
              "mother"=>"other",
              "father"=>"other",
              "son"=>"other",
              "daughter"=>"other",
              "sister"=>"other",
              "brother"=>"other",
              "friend"=>"other",
              "father-in-law"=>"other",
              "aunt"=>"other",
              "best friend"=>"other",
              "sister-in-law"=>"other",
              "son-in-law"=>"other",
              "daughter-in-law"=>"other",
              "burr"=>missing)
    v = [rr[x] for x in v]

    return v
end

function myparse(x)
    d = tryparse(Date, x, dateformat"mm/dd/yy")
    d = isnothing(d) ? missing : d
    d += Dates.Year(2000)
    return d
end

function load_demog(src)
    df = open(joinpath(pa, "roadmap_$(src).csv.gz")) do io
        CSV.read(io, DataFrame)
    end
    df = rename(df, "pt_rm_access_code"=>"pt_id", "cg_rm_access_code"=>"cg_id")
    id = vcat(df[:, :pt_id], df[:, :cg_id])
    n = size(df, 1)
    role = vcat(fill("pt", n), fill("cg", n))
    dx = DataFrame(ID=id, role=role)

    # Variables that differ between patient and caregiver
    for c in svars[src]
        dx[:, c] = vcat(df[:, "pt_$c"], df[:, "cg_$c"])
    end

    # Variables that are constant between two people in a dyad.
    for c in xvars[src]
        dx[:, c] = vcat(df[:, c], df[:, c])
    end

    dx = rename(dx, :age_enroll=>:age)
    dx[!, :age] = [ismissing(x) ? missing : Float64(x) for x in dx[:, :age]]

    dx[!, :cg_relationship] = recode_relationship(dx[:, :cg_relationship])

    dx = filter(r->r.gender != "Other", dx)

    if src == "bmt"
        dd = open(joinpath(pa, "RM_BMT_txp_discharge_dates.csv.gz")) do io
            CSV.read(io, DataFrame)
        end
        ddpt = dd[:, [:pt_rm_access_code, :bmt_date, :discharge_date]]
        ddpt = rename(ddpt, :pt_rm_access_code=>:ID)
        ddcg = dd[:, [:cg_rm_access_code, :bmt_date, :discharge_date]]
        ddcg = rename(ddcg, :cg_rm_access_code=>:ID)
        dd = vcat(ddpt, ddcg)
        dd[!, :bmt_date] = [myparse(x) for x in dd[:, :bmt_date]]
        dd[!, :discharge_date] = [myparse(x) for x in dd[:, :discharge_date]]
        dx = leftjoin(dx, dd, on=:ID)
    else
        dx[:, :bmt_date] .= missing
        dx[:, :discharge_date] .= missing
    end

    return dx
end

function load_demog2()
    bmt = load_demog("bmt")
    bmt[:, :study] .= "bmt"
    onc = load_demog("onc")
    onc[:, :study] .= "onc"
    onc[:, :cg_arm] .= "onc"
    return vcat(bmt, onc)
end

function load_mood_single(arm)

    da = open(GzipDecompressorStream, joinpath(pa, "long", "$(arm)_mood.csv.gz")) do io
        CSV.read(io, DataFrame)
    end

    da = da[:, [:Time, :id_patient, :id_caregiver, :mood1_patient, :mood1_caregiver]]

    d1 = da[:, [:Time, :id_patient, :mood1_patient]]
    d1 = rename(d1, :id_patient=>:id, :mood1_patient=>:mood1, :Time=>:date)
    d1 = d1[:, [:date, :id, :mood1]]
    d1[:, :role] .= "pt"

    d2 = da[:, [:Time, :id_caregiver, :mood1_caregiver]]
    d2 = rename(d2, :id_caregiver=>:id, :mood1_caregiver=>:mood1, :Time=>:date)
    d2 = d2[:, [:date, :id, :mood1]]
    d2[:, :role] .= "cg"

    dd = vcat(d1, d2)
    dd = dd[completecases(dd), :]
    dd = disallowmissing(dd)
    dd = dd[:, [:id, :date, :role, :mood1]]
    dd = sort(dd, [:id, :date])
    dd = filter(r->r.mood1 != 0, dd)

    return dd
end

function load_mood()

    d1 = load_mood_single("onc")
    d2 = load_mood_single("bmt")

    return vcat(d1, d2)
end

# Limit each dyad to the first 120 days of data
function limit_120!(da, dm)

    for j in 1:2
        da[j] = transform(groupby(da[j], :ID), x->1:size(x, 1))
    end
    @assert all(da[1][:, :x1] .== da[2][:, :x1])
    for j in 1:2
        ii = da[j][:, :x1] .<= 120
        da[j] = da[j][ii, :]
        dm[j] = dm[j][ii, :]
        da[j] = select(da[j], Not(:x1))
    end
end

# da[1], dm[1] are patients (id/date and HR)
# da[2], dm[2] are caregivers (id/date and HR)
function load_data(src)

    da, dm = [], []
    for x in ["pt", "cg"]
        f = @sprintf("%s_%s_wide.csv.gz", src, x)
        df = open(joinpath(pa, "wide", f)) do io
            CSV.read(io, DataFrame)
        end
        push!(da, df[:, 1:2])
        push!(dm, Matrix(df[:, 3:end]))
    end

    limit_120!(da, dm)

    return da, dm
end

function load_data2()
    bmta, bmtm = load_data("bmt")
    onca, oncm = load_data("onc")

    # da[1] is metadata for patients, da[2] is metadata for caregivers
    da = [vcat(bmta[1], onca[1]), vcat(bmta[2], onca[2])]

    # dm[1] is HR data for patients, dm[2] is HR data for caregivers
    dm = [vcat(bmtm[1], oncm[1]), vcat(bmtm[2], oncm[2])]

    # Create a day variable that counts from the first day
    for j in 1:2
        da[j] = transform(groupby(da[j], :ID), :Date=>x->minimum(x))
        da[j] = rename(da[j], :Date_function=>:FirstDate)
        da[j][:, :day] = da[j][:, :Date] - da[j][:, :FirstDate]
        da[j][!, :day] = [x.value for x in da[j][:, :day]]
    end

    return da, dm
end

function plot_means(rr, ifig)

    n = length(rr.Xm)
    ti = range(0, 24, n)

    f = Figure()
    ax = Axis(f[1, 1], xlabel="Time of day", ylabel="Mean heartrate",
              xticks=0:3:24, xlabelsize=18, ylabelsize=18)

    lines!(ti, rr.Xm, label="Patients")
    lines!(ti, rr.Ym, label="Caregivers")
    axislegend()
    save(@sprintf("plots/%03d.pdf", ifig), f)

    return ifig + 1
end

function get_season(mm)
    xn = coefnames(mm)
    c = coef(mm)
    ii = findall(xn .== "season_cos")[1]
    a = c[ii]
    jj = findall(xn .== "season_sin")[1]
    b = c[jj]
    amp = sqrt(a^2 + b^2)
    peak = atan(b, a)
    if peak < 0
        peak += pi
    end
    # Check second derivative for concavity
    if -a*cos(peak) - b*sin(peak) > 0
        peak += pi
    end
    peak *= 365.25 / (2 * pi)
    return amp, peak
end

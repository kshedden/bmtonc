using DataFrames, Dates, Printf

# Produce some basic summary statistics for each dataset

include("factor_setup.jl")

out = open("summary_stats.txt", "w")

for src in ["bmt", "onc"]

    mx, di = factor_setup(src)

    write(out, "$(src)\n")
    ndyad = length(unique(di[:, :ID]))
    write(out, "$(ndyad) dyads\n")

    nday = combine(x -> size(x, 1), groupby(di, :ID))
    mnm = minimum(nday[:, :x1])
    mxm = maximum(nday[:, :x1])
    write(out, "$(mnm)-$(mxm) days per dyad\n")

    ndd = size(di, 1)
    write(out, "$(ndd) dyad days\n")
    npd = size(mx, 2)
    write(out, "$(npd) person days\n")
    write(out, "")

    mdd = ndd / ndyad
    write(out, @sprintf("%.2f mean days per dyad\n", mdd))

    mpd = [count(x -> !ismissing(x), col) for col in eachcol(mx)]
    mpdmn = mean(mpd)
    mpdmd = median(mpd)
    write(out, @sprintf("Mean minutes per day: %.2f\n", mpdmn))
    write(out, @sprintf("Median minutes per day: %.2f\n\n", mpdmd))

end

close(out)

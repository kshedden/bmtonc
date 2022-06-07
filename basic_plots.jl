using CodecZlib, CSV, DataFrames, PyPlot, Printf, Dates

rm("plots", recursive = true, force = true)
mkdir("plots")

src = "bmt"

f = "/home/kshedden/data/Sung_Choi/long/$(src).csv.gz"
df = open(GzipDecompressorStream, f) do io
    CSV.read(io, DataFrame)
end
error("")
df[:, :Date] = Date.(df[:, :Time])

ifig = 0

function make_plots(ifig)

    for dd in groupby(df, :id_caregiver)
        for (jj, de) in enumerate(groupby(dd, :Date))

            # Only plot 10 days per dyad
            if jj > 5
                break
            end

            # Ids
            cg = first(de[:, :id_caregiver])
            pt = first(de[:, :id_patient])

            PyPlot.clf()
            ax = PyPlot.axes([0.1, 0.12, 0.8, 0.82])
            PyPlot.grid(true)

            du = de[:, [:Time, :HR_caregiver]]
            du = du[completecases(du), :]
            if size(du, 1) > 0
                PyPlot.plot(
                    du[:, :Time],
                    du[:, :HR_caregiver],
                    alpha = 0.5,
                    rasterized = true,
                    label = "Caregiver ($(cg))",
                )
            end

            du = de[:, [:Time, :HR_patient]]
            du = du[completecases(du), :]
            if size(du, 1) > 0
                PyPlot.plot(
                    du[:, :Time],
                    du[:, :HR_patient],
                    alpha = 0.5,
                    rasterized = true,
                    label = "Patient ($(pt))",
                )
            end

            for x in ax.get_xticklabels()
                x.set_rotation(30)
            end

            ha, lb = ax.get_legend_handles_labels()
            leg = PyPlot.figlegend(ha, lb, "upper center", ncol = 2)
            leg.draw_frame(false)
            PyPlot.ylabel("Heart rate")

            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    return ifig
end

ifig = make_plots(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=basic_plots_$(src).pdf $f`
run(c)

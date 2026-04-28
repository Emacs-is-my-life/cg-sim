# Plot simulation time vs memory size from the sweep results.
# Usage: gnuplot scripts/plot_results.gp
#
# Column layout of tmp/results/result_total.dat:
#   1: memory_MB   2: success (1/0)   3: simulation_time_us   4: peak_memory_KB
#
# Failed runs (success == 0) are filtered out so the line only shows real data.

set output 'tmp/results/result_total.png'

set title  "FlexInfer: Simulation Time vs Memory Size"
set xlabel "Memory Size (MB)"
set ylabel "Simulation Time (s)"

# Convert microseconds to seconds for a readable y-axis; drop failed rows.
plot 'tmp/results/result_total.dat' \
     using 1:($2 == 1 ? $3/1e6 : 1/0) \
     with linespoints lw 2 pt 7 ps 0.8 lc rgb "#1f77b4" title "FlexInfer"

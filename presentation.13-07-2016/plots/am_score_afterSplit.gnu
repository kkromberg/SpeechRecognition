set terminal postscript enhanced color font 'Helvetica,20'
set xlabel 'Number of densities per mixture'
set ylabel 'Averaged log-likelihood of training set'
set xtics 1.0
set ytics 5.0
set grid 
set size 1.0,1.0
set key right top

set style data linespoints
set yrange [25:40]

#set title "Log-likelihood of GMM models with expectation-maximization"
set output '| ps2pdf - plot.pdf'
set key autotitle columnhead
plot 'am_score.afterSplit' using 1:2 with line title 'Global pooling (max. approx.)'  linewidth 1 lt 1 lc rgb "red", \
     'am_score.afterSplit' using 1:3 with line title 'Mixture pooling (max. approx.)' linewidth 1 lt 1 lc rgb "blue", \
     'am_score.afterSplit' using 1:4 with line title 'No pooling (max. approx.)'      linewidth 1 lt 1 lc rgb "dark-green"

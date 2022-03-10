set style data lines
set term postscript enhanced color eps "Times" 28 dashlength 3

set xtics font "Times,24"
set ytics font "Times,24"
set format y "10^{%T}";

set style line 1 lw 5 lc rgbcolor "#000000" lt 1
set style line 2 lw 5 lc rgbcolor "#FF0000" lt 1
set style line 3 lw 5 lc rgbcolor "#0000FF" lt 1
set style line 4 lw 5 lc rgbcolor "#FF00FF" lt 1
set style line 5 lw 5 lc rgbcolor "#888888" lt 1
set style line 6 lw 5 lc rgbcolor "#FF5555" lt 1
set style line 7 lw 5 lc rgbcolor "#008800" lt 1
set style line 8 lw 5 lc rgbcolor "#FF00FF" lt 1

set xlabel "N" font "Times,28"
set ylabel "error" font "Times,28"

set xr [0:100]
set log y

set output "taylor_sin.eps"
plot "taylor_sin.out" u 1:2 ls 1 t "k=1",\
     "taylor_sin.out" u 1:3 ls 1 t "k=5",\
     "taylor_sin.out" u 1:4 ls 1 t "k=10",\
     "taylor_sin.out" u 1:5 ls 1 t "k=15"

set auto x
set log x
set format x "10^{%T}";
set format y "10^{%T}";

set output "pi_summation_slow.eps"
plot "pi_summation_slow.out" u 1:2 ls 1 notitle
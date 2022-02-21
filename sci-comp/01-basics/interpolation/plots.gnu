# use
# convert -colorspace sRGB -density 600 piecewise.pdf -background white -flatten -resize 800x -units pixelsperinch -density 224.993 piecewise.png

set style data lines
set term postscript enhanced color eps "Times" 28 dashlength 3

set xtics font "Times,24"
set ytics font "Times,24"

set style line 1 lw 3 lc rgbcolor "#000000" lt 1
set style line 2 lw 5 lc rgbcolor "#FF0000" lt 1
set style line 3 lw 5 lc rgbcolor "#0000FF" lt 1
set style line 4 lw 5 lc rgbcolor "#FF00FF" lt 1
set style line 5 lw 5 lc rgbcolor "#888888" lt 1
set style line 6 lw 5 lc rgbcolor "#FF5555" lt 1
set style line 7 lw 5 lc rgbcolor "#008800" lt 1
set style line 8 lw 5 lc rgbcolor "#FF00FF" lt 1

set xlabel "x" font "Times,28"
set ylabel "f(x)" font "Times,28"

set xr [0:7]

unset key

set output "piecewise.eps"
plot "func_exact.out" u 1:2 w l ls 1,\
     "func_exact.out" u 1:3 w l ls 2,\
     "func_exact20.out" u 1:3 w l ls 3,\
     "samples10.out" u 1:2 w p lc rgbcolor "#FF0000" pt 13 ps 3,\
     "samples20.out" u 1:2 w p lc rgbcolor "#0000FF" pt 6 ps 2

set output "linear.eps"
plot "func_exact.out" u 1:2 ls 1,\
     "func_exact.out" u 1:4 ls 2,\
     "func_exact20.out" u 1:4 ls 3,\
     "samples10.out" u 1:2 w p lc rgbcolor "#FF0000" pt 13 ps 3,\
     "samples20.out" u 1:2 w p lc rgbcolor "#0000FF" pt 6 ps 2

set yr [-2:2]
set output "piecewise_error.eps"
plot "func_exact.out" u 1:($3-$2) ls 2,\
     "func_exact20.out" u 1:($3-$2) ls 3

set yr [-0.4:0.4]
set output "linear_error.eps"
plot "func_exact.out" u 1:($4-$2) ls 2,\
     "func_exact20.out" u 1:($4-$2) ls 3
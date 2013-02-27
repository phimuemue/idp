set style line 1  lc rgb '#0025ad' lt 1 lw 1.5
set style line 2  lc rgb '#00ada4' lt 1 lw 1.5
set style line 3  lc rgb '#09ad00' lt 1 lw 1.5

set term pdf
set output "simple_parametric_2d.pdf"
unset key

set label 1 "a*x+t" at 8.8,35.8 front center
unset label 2
unset label 3
unset label 4

set colorbox user horizontal origin 0.73,0.235 size 0.18,0.035 front
set cbrange [1:3]
set cbtics ('a=1' 1,'a=3' 3) offset 0,0.5 scale 0
set palette defined (1  '#0025ad', 6  '#00ada4', 12 '#09ad00')

set xlabel "x"
set ylabel "y"

plot for [a=1:3] a*x+1 w lines ls (a), 1/0 w image

clear 
reset
set palette rgb 7,5,15
unset colorbox
set output "simple_parametric_3d.pdf"
set ylabel "a"
set yrange [1:3]
set pm3d
splot y*x+1 

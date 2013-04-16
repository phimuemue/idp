set term pdf
set output "asdf.pdf" 

# define so that only triangle area is plotted (otherwise undefined)
triangle(x,y) = log(sgn((x+y-1)*-1))+1

# first order basis functions
p1(x,y) = 1-2*y
p2(x,y) = -1 + 2*x + 2*y
p3(x,y) = 1-2*x
h2 = 10
h3 = 10

# f is the singularity function we want to plot (without log!)
#f(x,y) = (20*y**2+(40*x-30)*y+20*x**2-30*x)/(2*y**2+(4*x-3)*y+2*x**2-3*x+1)
f(x,y) = (p2(x,y)*h2+p3(x,y)*h3)/(-p1(x,y))

# g applies log
g(x,y) = log(f(x,y))/log(10)

# adjust plot axes, legend, etc.
unset key
set xlabel "x"
set ylabel "y"
set view map
set xrange [0:1]
set yrange [0:1]
set cbrange [0:6]
set isosamples 100
set samples 1000

set style line 1 lt 0 lw 0 pt 0 linecolor rgb "black"

# colors
set pm3d
set palette model RGB
#set palette model RGB defined (-4 "green", 0.7 "dark-green", 0.7 "yellow", 1.17 "dark-yellow", 1.17 "red", 5 "dark-red" )
set palette model RGB defined (0 "green", 0.7 "dark-green", 0.7 "red", 1.2 "dark-red", 1.2 "green", 6 "dark-green" )
set cbtics 0
set cbtics add ("1" 0)
set cbtics add ("5" 0.7)
set cbtics add ("15" 1.2)
set cbtics add ("10²" 2)
set cbtics add ("10³" 3)
set cbtics add ("10⁴" 4)
set cbtics add ("10⁵" 5)
set cbtics add ("10⁶" 6)

#set palette model RGB defined (1.17 "red", 6 "dark-red" )

#splot 1.1 with image
splot g(x,y)*triangle(x,y) ls 1

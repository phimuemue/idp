set term pdf
set output "asdf.pdf" 

# define so that only triangle area is plotted (otherwise undefined)
triangle(x,y) = log(sgn((x+y-1)*-1))+1

# f is the singularity function we want to plot (without log!)
f(x,y) = (20*y**2+(40*x-30)*y+20*x**2-30*x)/(2*y**2+(4*x-3)*y+2*x**2-3*x+1)
f(x,y) = (1-2*x+24*y)/(-1+2*y)

# g applies log
g(x,y) = log(f(x,y))/log(10)

# adjust plot axes, legend, etc.
unset key
set xlabel "x"
set ylabel "y"
set view map
set xrange [0:1]
set yrange [0:1]
set isosamples 100
set samples 1000

# colors
set pm3d
set palette model RGB
set palette model RGB defined (-4 "green", 0.7 "dark-green", 0.7 "yellow", 1.17 "dark-yellow", 1.17 "red", 5 "dark-red" )
#set palette model RGB defined (1.17 "red", 6 "dark-red" )

splot g(x,y)*triangle(x,y) with image

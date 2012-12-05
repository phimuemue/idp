# TODO:
# * make main routine such that it accepts a variable number of functions, plotting each one in a single gnuplot window
# * correct order of variables
# * possibility to choose domain for plot variables/parameters dynamically
# * could be nice: possibility to send custom command to (all instances of) gnuplot

from numpy import *
import Gnuplot, Gnuplot.funcutils
import pygtk
import gtk
import re
import optparse
import time
import os.path
import os
import glob

class SimplePlotter:
    """This class encapsules a whole bunch of things useful for "interactive" 
    gnuplotting. Amongst others:
        * self.window: Reference to the window used for interaction
        * self.sliders: Sliders for variables
        * self.xratios/self.yradios: Which variables to use as axes?
        * self.variables: Names of the variables
        * self.func: an array of the functions in their raw form
        * self.xyfunc: an array of the functions, x/y-coords already properly substituted"""

    def delete_event(self, widget, event, data=None):
        """Things to do when window is deleted."""
        return False

    def destroy(self, widget, data=None):
        """Things to do when window is destroyed."""
        if os.path.exists(self.foldername):
            texfile = open(self.foldername + "/tex.tex", "w")
            texfile.write("%% %d variables in here:\n" % len(self.variables))
            vars_values = zip(self.variables, [s.get_value() for s in self.sliders])
            vars_values = map(lambda x: "%s = %s"%(x[0], str(x[1])), vars_values)
            print vars_values
            texfile.write("% " + ", ".join(vars_values)+"\n")
            texfile.write("\\begin{figure}[ht]\n")
            texfile.write("\\centering\n")
            for f in glob.glob(self.foldername+"/*.%s"%self.img_extension):
                print f
                texfile.write("  \\subfigure[] {\n")
                texfile.write("    \\includegraphics[scale=\zoomfactor]{{{%s}}}\n"%f[:-(len(self.img_extension)+1)])
                texfile.write("  }\n")
            texfile.write("\\caption{}\n")
            texfile.write("\\label{}\n")
            texfile.write("\\end{figure}\n")
        gtk.main_quit()

    def redraw(self, adj, data=None):
        """This is invoked to generate a plot for *all* given functions.
        Delegates the work to the function drawspecific."""
        self.adjustvars()
        for i in xrange(len(self.func)):
            if self.plotcommand.startswith("plot"):
                # collect proper variables
                avg_vars = [s.get_value() for (_i,s) in enumerate(self.sliders)
                            if self.variables[_i].lower().startswith(self.xvar[0].lower())]
                avg_vars = float(sum(avg_vars))/len(avg_vars)
                print  "set arrow 1 from %f,-100000 to %f,100000"%(avg_vars, avg_vars)
                self.gnuplot[i]("set arrow 1 from %f,-100000 to %f,100000"%(avg_vars, avg_vars))
            self.gnuplot[i]('replot')

    def adjustvars(self):
        """Tells gnuplot to set the variables/parameters."""
        for i in xrange(len(self.func)):
            for (varname, varval) in zip(self.variables, [s.get_value() for s in self.sliders]):
                self.gnuplot[i]('%s=%f' % (varname, varval))

    def adjuststuff(self, widget, data=None):
        """Determines which variables are plot-axes-variables and wich
        ones are parameters. Decides whether to use a 2D- or a 3D-plot."""
        # numerical integration by streifensumme
        stripe_amount = int(self.stripes_spin.get_value()) 
        stripe_width = 1./stripe_amount
        stripe_midpoints = [1./(2*stripe_amount) + i*stripe_width for i in xrange(stripe_amount)]
        self.tfunc = []
        for f in self.func:
            tmp = ["+abs((%s)*(%f))"%(f.replace("x",str(midpoint)), stripe_width)
                   for midpoint in stripe_midpoints]
            tmp = "".join(tmp)
            self.tfunc.append(tmp)
        # determine variables
        subs = []
        xvar = self.variables[0]
        yvar = self.variables[0]
        for (i,r) in enumerate(self.xradios):
            if r.get_active():
                xvar = self.variables[i]
        for (i,r) in enumerate(self.yradios):
            if r.get_active():
                yvar = self.variables[i]
        subs.append((xvar, "x"))
        subs.append((yvar, "y"))
        self.plotcommand = "splot " if xvar!=yvar else "plot "
        # fill xyfunc with data for gnuplot
        self.xyfunc = []
        for f in self.tfunc:
            tmp = f
            for (s,t) in subs:
                tmp = tmp.replace(s,t)
            self.xyfunc.append(tmp)
        # adjust gnuplot properly
        for i in xrange(len(self.func)):
            gp = self.gnuplot[i]
            if xvar != yvar:
                gp("set xlabel \"%s\""%xvar)
                gp("set ylabel \"%s\""%yvar)
                gp("set xrange " + self.urange)
                gp("set yrange " + self.hrange)
                gp("set pm3d")
            else:
                gp("unset ylabel")
                gp("set xlabel \"%s\""%xvar)
                gp("set autoscale")
                gp("unset pm3d")
            self.adjustvars()
            gp('%s %s ls 1 title "%d. Function"' % (self.plotcommand, self.xyfunc[i], i))

    def adjustment_from_range(self,r):
        """Returns a gtk.Adjustment from a "gnuplot"-range.
        A gnuplot-range has the format [-3:67]."""
        _r = r[1:-1]
        _r = _r.split(":")
        l = float(_r[0])
        h = float(_r[1])
        return gtk.Adjustment((l+h)/2., l, h, step_incr=0.5)

    def do_settings(self, widget):
        """Determines the settings and applies them (these may be varied 
        for efficiency)."""
        # determine plotting vars
        assert(len(self.variables)==len(self.sliders))
        for gp in self.gnuplot:
            gp("set samples %d"%self.samples_spin.get_value())
            gp("set isosamples %d"%self.isosamples_spin.get_value())
            gp("set xrange [%d:%d]"%(self.x_lower_spin.get_value(), self.x_upper_spin.get_value()))
            gp("set yrange [%d:%d]"%(self.y_lower_spin.get_value(), self.y_upper_spin.get_value()))
            gp("replot")

    def init_settings_page(self):
        """Fills the settings page."""
        # samples
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Samples: "))
        self.samples_spin = gtk.SpinButton(gtk.Adjustment(100,2,1000,1))
        hbox.pack_start(self.samples_spin)
        self.settings_box.pack_start(hbox)
        self.samples_spin.connect("value_changed", self.do_settings) 
        # isosamples
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Isosamples: "))
        self.isosamples_spin = gtk.SpinButton(gtk.Adjustment(10,2,1000,1))
        hbox.pack_start(self.isosamples_spin)
        self.settings_box.pack_start(hbox)
        self.isosamples_spin.connect("value_changed", self.do_settings) 
        # plotting boundaries
        # x/u-component
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("lower x: "))
        self.x_lower_spin = gtk.SpinButton(gtk.Adjustment(-4,-500,500,1))
        hbox.pack_start(self.x_lower_spin)
        self.settings_box.pack_start(hbox)
        self.x_lower_spin.connect("value_changed", self.do_settings) 
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("upper x: "))
        self.x_upper_spin = gtk.SpinButton(gtk.Adjustment(4,-500,500,1))
        hbox.pack_start(self.x_upper_spin)
        self.settings_box.pack_start(hbox)
        self.x_upper_spin.connect("value_changed", self.do_settings) 
        # y/h-component
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("lower y: "))
        self.y_lower_spin = gtk.SpinButton(gtk.Adjustment(8,-500,500,1))
        hbox.pack_start(self.y_lower_spin)
        self.settings_box.pack_start(hbox)
        self.y_lower_spin.connect("value_changed", self.do_settings) 
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("upper y: "))
        self.y_upper_spin = gtk.SpinButton(gtk.Adjustment(12,-500,500,1))
        hbox.pack_start(self.y_upper_spin)
        self.settings_box.pack_start(hbox)
        self.y_upper_spin.connect("value_changed", self.do_settings) 
        # amount of stipes for streifenmethode numerical integration
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Stripes: "))
        self.stripes_spin = gtk.SpinButton(gtk.Adjustment(5,1,1000,1))
        hbox.pack_start(self.stripes_spin)
        self.settings_box.pack_start(hbox)
        self.stripes_spin.connect("value_changed", self.adjuststuff) 

    def init_variables_page(self):
        """Creates sliders and radio buttons for the single variables."""
        # create elements for variables!!
        self.variables = list(set(re.findall(
                    "[a-zA-Z_]+[1-9]*",self.func[0] +"+"+ self.func[1])))
        self.variables = filter(lambda x:x!="x", self.variables)
        def filter_func(x):
            if x in ["abs", "sin", "cos", "log", "tan", "e", "ln"]:
                return False
            if re.match("^e[0-9]*$", x) is not None:
                return False
            return True
        self.variables = filter(filter_func, self.variables)
        print self.variables
        # an auxiliary function to sort the variables properly
        def varkey(x):
            tmp = x.split("_")
            print "Varname: " + x
            print tmp
            return (int(tmp[1]),255-ord(tmp[0]))
        self.variables.sort(key=varkey)
        self.sliders = []
        self.xradios = []
        self.yradios = []
        for v in self.variables:
            # a whole container for each variable is necessary
            newvbox = gtk.VBox()
            self.variable_hbox.add(newvbox)
            # description
            newvbox.pack_start(gtk.Label(v), False, False)
            # options for x
            group = None if len(self.xradios)==0 else self.xradios[0]
            newx = gtk.RadioButton(group, "")
            self.xradios.append(newx)
            newvbox.pack_start(newx, False, False)
            newx.connect("toggled", self.adjuststuff)
            # options for x
            group = None if len(self.yradios)==0 else self.yradios[0]
            newy = gtk.RadioButton(group, "")
            self.yradios.append(newy)
            newvbox.pack_start(newy, False, False)
            newy.connect("toggled", self.adjuststuff)
            # slider
            newadjustment = self.adjustment_from_range(self.urange)
            if v[0].lower() == "h":
                newadjustment = self.adjustment_from_range(self.hrange)
            newadjustment.connect("value_changed", self.redraw)
            self.sliders.append(gtk.VScale(newadjustment))
            self.sliders[-1].set_size_request(10,200)
            self.sliders[-1].set_inverted(True)
            newvbox.add(self.sliders[-1])
        self.xvar = self.variables[0]
        self.yvar = self.variables[0]

    def export_image_with_terminal(self, gp, term, filename, i):
        gp("set style line 1 linecolor rgb \"black\"")
        if term=="jpg" or term=="jpeg":
            size = "1024,768"
            gp("set term jpeg size %s truecolor"%size)
            gp("set output \"%s/%sf%d.jpg\""%(self.foldername,filename,i))
        if term=="pdf":
            gp("set term pdfcairo size 5.0in,3.0in")
            gp("set output \"%s/%sf%d.pdf\""%(self.foldername,filename,i))
        gp("replot")
        gp("set term wxt")
            

    def on_export(self, widget):
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        filename = ""
        tmp = [str(s.get_value()) for s in self.sliders]
        for (i,r) in enumerate(self.xradios):
            if r.get_active():
                tmp[i] = "x"
        for (i,r) in enumerate(self.yradios):
            if r.get_active():
                tmp[i] = "y"
            filename = "_".join(tmp) 
        for (i,gp) in enumerate(self.gnuplot):
            self.export_image_with_terminal(gp, self.img_extension, filename, i)
        print "Exporting!"

    def init_gtk_layout(self):
        """Creates gtk stuff: Window, notebook. And shows the window."""
        # toplevel window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("delete_event", self.delete_event)
        self.window.connect("destroy", self.destroy)
        self.window.set_border_width(10)
        # user interface stuff!
        self.vbox = gtk.VBox()
        self.window.add(self.vbox)
        self.notebook = gtk.Notebook()
        self.vbox.add(self.notebook)
        self.variable_hbox = gtk.HBox()
        self.settings_box = gtk.VBox()
        self.notebook.append_page(self.variable_hbox, gtk.Label("Variables/Parameters"))
        self.notebook.append_page(self.settings_box, gtk.Label("Settings"))
        self.window.add(self.variable_hbox)
        # export button
        vbox = gtk.VBox()
        self.vbox.add(vbox)
        button = gtk.Button("Export")
        vbox.add(button)
        button.connect("clicked", self.on_export)
        # initialize single pages of the notebook
        self.init_variables_page()
        self.init_settings_page()
        self.window.show_all()

    def __init__(self, func1="h[1]+h[2]*h[3]", func2="h[1]+h[2]", urange="[-5:5]", hrange="[8:12]"):
        """Initialization of the window."""
        # Configure Gnuplot settings
        self.urange = urange
        self.hrange = hrange
        self.plotcommand = "plot"
        self.func = [func1, func2]
        # initialize gnuplot for me
        self.gnuplot = []
        for f in self.func:
            newgp = Gnuplot.Gnuplot(debug=0)
            self.gnuplot.append(newgp)
            self.gnuplot[-1]("set xrange " + urange)
            self.gnuplot[-1]("set yrange " + hrange)
            self.gnuplot[-1]("set pm3d")
        self.foldername = time.strftime("%Y%m%d%H%M%S")
        self.img_extension = "pdf"
        # Initialize Gtk layout
        self.init_gtk_layout()
        self.adjuststuff(None)

    def main(self):
        """Just fires the gtk main loop."""
        gtk.main()

def prettify_function(f):
    """Brings functions from maple to gnuplot-syntax."""
    result = f
    result = re.sub("([a-z_]*)\[(.*?)\]", "\g<1>_\g<2>", result) 
    result = result.replace("ln", "log")
    result = result.replace("^","**")
    return result

def make_range(start, end):
    """Returns a string indicating a range that can be given to gnuplot."""
    return "[%d:%d]" % (start, end)

def get_range(range):
    """Takes a gnuplot-style range and returns a tuple (lower bound/upper bound)."""
    result = re.match('\[(.*?):(.*?)\]', range)
    return result.group(1), result.group(2)

def read_file(path):
    """Reads a file generated by maple containing two terms."""
    result = []
    for line in open(path,"r"):
        if line[0]==">":
            result.append("")
        else:
            result[-1] = result[-1] + line.translate(None,"\"\\\n ")
    return result

            

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--urange", default="[-5:5]")
    parser.add_option("--hrange", default="[8:12]")
    parser.add_option("-f", "--file", default="norm_stuff.txt")
    opts, args = parser.parse_args()
    stuff = read_file(opts.file)
    simpleplotter = SimplePlotter(
                                  prettify_function(stuff[0]), prettify_function(stuff[1]),
                                  urange=opts.urange, hrange=opts.hrange
            )
    simpleplotter.main()

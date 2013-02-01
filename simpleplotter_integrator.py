# TODO:
# * possibility to choose domain for plot variables/parameters dynamically

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

from mathparser import Plotter

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
        gtk.main_quit()
        return

    def redraw(self, adj, data=None):
        """This is invoked to generate a plot for *all* given functions.
        Delegates the work to the function drawspecific."""
        self.adjustvars()
        self.adjust_and_plot(adj)

    def adjustvars(self):
        """Tells gnuplot to set the variables/parameters."""
        print "Setting variables!!!"
        self.plotter.setvars(*zip(self.variables, [s.get_value() for s in self.sliders]))

    def adjust_and_plot(self, widget, data=None, two_dim=True):
        """Determines which variables are plot-axes-variables and wich
        ones are parameters. Decides whether to use a 2D- or a 3D-plot."""
        # numerical integration by streifensumme
        stripe_amount = int(self.stripes_spin.get_value())
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
        self.plotter.setvars(*zip(self.variables, [s.get_value() for s in self.sliders]))
        if xvar != yvar:
            self.plotter.gp('set xlabel "%s"' % xvar)
            self.plotter.gp('set ylabel "%s"' % yvar)
            self.plotter.gp("set pm3d")
        else:
            self.plotter.gp("unset ylabel")
            self.plotter.gp("set xlabel \"%s\"" % xvar)
            self.plotter.gp("set autoscale")
            self.plotter.gp("unset pm3d")
        self.plotter.replot((xvar, yvar))
        return

    def adjustment_from_range(self, r):
        """Returns a gtk.Adjustment from a "gnuplot"-range.
        A gnuplot-range has the format [-3:67]."""
        return gtk.Adjustment((r[0]+r[1])/2., r[0], r[1], step_incr=0.5)

    def do_settings(self, widget):
        """Determines the settings and applies them (these may be varied
        for efficiency)."""
        # determine plotting vars
        assert(len(self.variables)==len(self.sliders))
        self.plotter.gp("set samples %d" % self.samples_spin.get_value())
        self.plotter.gp("set isosamples %d" % self.isosamples_spin.get_value())
        self.plotter.gp("set xrange [%d:%d]" % (self.x_lower_spin.get_value(), self.x_upper_spin.get_value()))
        self.plotter.gp("set yrange [%d:%d]" % (self.y_lower_spin.get_value(), self.y_upper_spin.get_value()))
        self.adjust_and_plot()

    def init_plottings_page(self):
        """Initialize page that holds all auxiliary and plotted funcs.
        This is basically just beautification."""
        self.tv_plottings = gtk.TreeView()
        tv = self.tv_plottings
        sw = gtk.ScrolledWindow()
        self.plottings_box.pack_start(sw)
        sw.add(tv)
        self.plottings_model = gtk.ListStore(str, str)
        model = self.plottings_model
        tv.set_model(self.plottings_model)
        tv.append_column(gtk.TreeViewColumn("Name", gtk.CellRendererText(), text = 0))
        tv.append_column(gtk.TreeViewColumn("Term", gtk.CellRendererText(), text = 1))
        tv.set_headers_visible(True)
        for (a,b) in self.auxiliaries:
            model.append([a, b])
        model.append(["---","-------------"]);
        for (i,f) in enumerate(self.func):
            model.append(["%d. func"%i, str(f)])

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
        self.stripes_spin.connect("value_changed", self.adjust_and_plot)

    def init_variables_page(self):
        """Creates sliders and radio buttons for the single variables."""
        print "Checking variables 'n stuff"
        # create elements for variables!!
        self.variables = list(set(re.findall(
                    r"([a-zA-Z_]+\d*\b)(?:[^(]|$)", "+".join(self.func + ([aux[1] for aux in self.auxiliaries])))))
        #print self.variables
        self.variables = filter(lambda x:x not in["x","y"], self.variables)
        #print "Auxiliaries: " + str(self.auxiliaries)
        def is_not_auxiliary(x):
            aux_l = [a[0] for a in self.auxiliaries]
            for au in aux_l:
                if au.startswith(x):
                    return False
            return True
        self.variables = filter(is_not_auxiliary, self.variables)
        def filter_func(x):
            if x in ["abs", "sin", "cos", "log", "tan", "e", "ln"]:
                return False
            if re.match("^e[0-9]*$", x) is not None:
                return False
            return True
        self.variables = filter(filter_func, self.variables)
        #print "Variables: " + str(self.variables)
        # an auxiliary function to sort the variables properly
        # this was earlier more complex, but now it's easy!
        def varkey(x):
            return x
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
            newx.connect("toggled", self.adjust_and_plot)
            # options for x
            group = None if len(self.yradios)==0 else self.yradios[0]
            newy = gtk.RadioButton(group, "")
            self.yradios.append(newy)
            newvbox.pack_start(newy, False, False)
            newy.connect("toggled", self.adjust_and_plot)
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

    def on_export(self, widget):
        foldername = time.strftime("%Y-%m-%d-%H-%M-%S")
        try:
            os.makedirs(foldername)
        except OSError:
            print('Simpleplotter warning: Folder %s already exists.' % foldername)
        for name, parser in self.plotter.parsers.items():
            filename = name
            parser.gp("set term pdfcairo size 5.0in,3.0in")
            parser.gp("set output \"%s/%s.pdf\"" % (foldername, filename))
            parser.gp("replot")
            parser.gp("set term wxt")
        texfile = open(foldername + "/tex.tex", "w")
        texfile.write("%% %d variables in here:\n" % len(self.variables))
        vars_values = zip(self.variables, [s.get_value() for s in self.sliders])
        vars_values = map(lambda x: "%s = %s"%(x[0], str(x[1])), vars_values)
        #print vars_values
        texfile.write("% " + ", ".join(vars_values)+"\n")
        texfile.write("\\begin{figure}[ht]\n")
        texfile.write("\\centering\n")
        while len(glob.glob(foldername+"/*.pdf")) < len(self.plotter.parsers):
            pass
        for f in glob.glob(foldername+"/*.pdf"):
            print "File: " + f
            texfile.write("  \\subfigure[] {\n")
            texfile.write("    \\includegraphics[scale=\zoomfactor]{{{%s}}}\n"%f[:-4])
            texfile.write("  }\n")
        texfile.write("\\caption{}\n")
        texfile.write("\\label{}\n")
        texfile.write("\\end{figure}\n")
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
        self.plottings_box = gtk.VBox()
        self.notebook.append_page(self.variable_hbox, gtk.Label("Variables/Parameters"))
        self.notebook.append_page(self.settings_box, gtk.Label("Settings"))
        self.notebook.append_page(self.plottings_box, gtk.Label("Plottings"))
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
        self.init_plottings_page()
        self.window.show_all()

    def __init__(self, functions, auxiliaries, urange=(-5,5), hrange=(8,12)):
        """Initialization of the window.
            * funcitons: List of strings representing stuff to be plotted
            * auxiliaries: List of strings of the form "varname=varcomputationstuff" ("intermediate vars")
        """
        # Configure Gnuplot settings
        self.urange = urange
        self.hrange = hrange
        self.plotcommand = "plot"
        self.func = functions
        self.auxiliaries = auxiliaries

        # Read functions
        tmp = [map(lambda x:x[0]+"="+x[1], self.auxiliaries) + ["plotfnc%d(x,y)=%s"%(i,c)] for (i,c) in enumerate(self.func)]
        tmp = map(lambda x: ", ".join(x), tmp)

        # Initialize plotters
        self.plotter = Plotter(tmp, integrate=True)
        self.plotter.gp("set style line 1 linecolor rgb \"black\"")
        #self.plotter.gp('set xrange %s' % urange)
        #self.plotter.gp('set yrange %s' % hrange)
        #self.plotter.gp('set pm3d')

        # Export settings

        # Initialize Gtk layout
        self.init_gtk_layout()

        self.adjustvars()
        self.plotter.plot((self.variables[0],))

        # first round!
        self.adjust_and_plot(None)

    def main(self):
        """Just fires the gtk main loop."""
        gtk.main()

def prettify_function(f):
    """Brings functions from maple to gnuplot-syntax."""
    result = f
    result = re.sub("([a-z_]*)\[(.*?)\]", "\g<1>_\g<2>", result)
    result = result.replace("ln", "log")
    result = result.replace("^", "**")
    return result

def read_file(path):
    """Reads a file generated by maple and converted by our script containing two terms."""
    functions = []
    auxiliaries = []
    for line in open(path,"r"):
        if "=" in line:
            varname, content = line.strip().split("=")
            auxiliaries.append([varname, prettify_function(content)])
        else:
            functions.append(prettify_function(line.strip()))
    return (functions, auxiliaries)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--urange", default=(-5,5))
    parser.add_option("--hrange", default=(8,12))
    parser.add_option("-f", "--file", default="norm_stuff.txt")
    opts, args = parser.parse_args()
    stuff = read_file(opts.file)
    functions, auxiliaries = read_file(opts.file)
    simpleplotter = SimplePlotter(
                                  functions, auxiliaries,
                                  urange=opts.urange, hrange=opts.hrange
            )
    simpleplotter.main()


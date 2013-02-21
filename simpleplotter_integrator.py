# TODO:
# * possibility to choose domain for plot variables/parameters dynamically

import gtk
import argparse
from time import strftime
from os import makedirs
from glob import glob

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

    def plot(self, widget=None, data=None):
        """Determines which variables are plot-axes-variables and wich
        ones are parameters. Decides whether to use a 2D- or a 3D-plot."""
        if isinstance(widget, gtk.RadioButton):
            if not widget.get_active():
                return
            else:
                self.plotvars = (widget.var, self.plotvars[1]) if widget.index == 'x' else (self.plotvars[0], widget.var)
        # determine variables
        self.plots.setvars(*zip(self.variables, [s.get_value() for s in self.sliders]))
        self.plots.plot(self.plotvars)

    def do_settings(self, widget):
        """Determines the settings and applies them (these may be varied
        for efficiency)."""
        # determine plotting vars
        if widget == self.samples_spin:
            self.plots.gp("set samples %d" % self.samples_spin.get_value())
        elif widget == self.isosamples_spin:
            self.plots.gp("set isosamples %d" % self.isosamples_spin.get_value())
        elif widget == self.x_lower_spin or widget == self.x_upper_spin:
            self.plots.gp("set xrange [%d:%d]" % (self.x_lower_spin.get_value(), self.x_upper_spin.get_value()))
        elif widget == self.y_lower_spin or widget == self.y_upper_spin:
            self.plots.gp("set yrange [%d:%d]" % (self.y_lower_spin.get_value(), self.y_upper_spin.get_value()))
        elif widget == self.stripes_spin:
            self.plots.settings(intstops = int(self.stripes_spin.get_value()))
        self.plot(widget)

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

    def init_settings_page(self):
        """Fills the settings page."""
        # samples
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Samples: "))
        self.samples_spin = gtk.SpinButton(gtk.Adjustment(self.default_samples,2,1000,1))
        hbox.pack_start(self.samples_spin)
        self.settings_box.pack_start(hbox)
        self.samples_spin.connect("value_changed", self.do_settings)
        # isosamples
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Isosamples: "))
        self.isosamples_spin = gtk.SpinButton(gtk.Adjustment(self.default_isosamples,2,1000,1))
        hbox.pack_start(self.isosamples_spin)
        self.settings_box.pack_start(hbox)
        self.isosamples_spin.connect("value_changed", self.do_settings)
        # plotting boundaries
        # x/u-component
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("lower x: "))
        self.x_lower_spin = gtk.SpinButton(gtk.Adjustment(self.default_h_lower,-500,500,1))
        hbox.pack_start(self.x_lower_spin)
        self.settings_box.pack_start(hbox)
        self.x_lower_spin.connect("value_changed", self.do_settings)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("upper x: "))
        self.x_upper_spin = gtk.SpinButton(gtk.Adjustment(self.default_h_upper,-500,500,1))
        hbox.pack_start(self.x_upper_spin)
        self.settings_box.pack_start(hbox)
        self.x_upper_spin.connect("value_changed", self.do_settings)
        # y/h-component
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("lower y: "))
        self.y_lower_spin = gtk.SpinButton(gtk.Adjustment(self.default_u_lower,-500,500,1))
        hbox.pack_start(self.y_lower_spin)
        self.settings_box.pack_start(hbox)
        self.y_lower_spin.connect("value_changed", self.do_settings)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("upper y: "))
        self.y_upper_spin = gtk.SpinButton(gtk.Adjustment(self.default_u_upper,-500,500,1))
        hbox.pack_start(self.y_upper_spin)
        self.settings_box.pack_start(hbox)
        self.y_upper_spin.connect("value_changed", self.do_settings)
        # amount of stipes for streifenmethode numerical integration
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Stripes: "))
        self.stripes_spin = gtk.SpinButton(gtk.Adjustment(self.default_intstops,1,1000,1))
        hbox.pack_start(self.stripes_spin)
        self.settings_box.pack_start(hbox)
        self.stripes_spin.connect("value_changed", self.do_settings)

    def init_variables_page(self):
        """Creates sliders and radio buttons for the single variables."""
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
            newx.index = 'x'
            newx.var = v
            self.xradios.append(newx)
            newvbox.pack_start(newx, False, False)
            newx.connect("toggled", self.plot)
            # options for y
            group = None if len(self.yradios)==0 else self.yradios[0]
            newy = gtk.RadioButton(group, "")
            newy.index = 'y'
            newy.var = v
            self.yradios.append(newy)
            newvbox.pack_start(newy, False, False)
            newy.connect("toggled", self.plot)
            # slider
            if v[0].lower() == 'h':
                newadjustment = gtk.Adjustment(self.default_h, self.hrange[0], self.hrange[1], step_incr=0.5)
            elif v[0].lower() == 'u':
                newadjustment = gtk.Adjustment(self.default_u, self.urange[0], self.urange[1], step_incr=0.5)
            else:
                newadjustment = gtk.Adjustment(0.5, 0, 1, 0.1)
            newadjustment.connect("value_changed", self.plot)
            self.sliders.append(gtk.VScale(newadjustment))
            self.sliders[-1].set_size_request(10,200)
            self.sliders[-1].set_inverted(True)
            newvbox.add(self.sliders[-1])

    def on_export(self, widget):
        foldername = strftime("%Y-%m-%d-%H-%M-%S")
        try:
            makedirs(foldername)
        except OSError:
            print('Simpleplotter warning: Folder %s already exists.' % foldername)
        self.plots.export(foldername)
        texfile = open(foldername + '/tex.tex', 'w')
        texfile.write('%% %d variables in here:\n' % len(self.variables))
        vars_values = zip(self.variables, [s.get_value() for s in self.sliders])
        vars_values = map(lambda x: '%s = %s' % (x[0], str(x[1])), vars_values)
        #print vars_values
        texfile.write('% ' + ', '.join(vars_values)+'\n')
        texfile.write('\\begin{figure}[ht]\n')
        texfile.write('\\centering\n')
        while len(glob(foldername+'/*.pdf')) < len(self.plots.parsers):
            pass
        for f in glob(foldername+'/*.pdf'):
            texfile.write('  \\subfigure[] {\n')
            texfile.write('    \\includegraphics[scale=\zoomfactor]{{{%s}}}\n' % f[:-4])
            texfile.write('  }\n')
        texfile.write('\\caption{}\n')
        texfile.write('\\label{}\n')
        texfile.write('\\end{figure}\n')
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
        self.vbox.pack_start(self.notebook, expand=True, fill=True)
        self.variable_hbox = gtk.HBox()
        self.settings_box = gtk.VBox()
        self.plottings_box = gtk.VBox()
        self.notebook.append_page(self.variable_hbox, gtk.Label("Variables/Parameters"))
        self.notebook.append_page(self.settings_box, gtk.Label("Settings"))
        self.notebook.append_page(self.plottings_box, gtk.Label("Plottings"))

        # export button
        vbox = gtk.VBox()
        self.vbox.pack_end(vbox, expand=False, fill=False)
        button = gtk.Button("Export")
        vbox.pack_end(button, expand=False)
        button.connect("clicked", self.on_export)

        # initialize single pages of the notebook
        self.init_variables_page()
        self.init_settings_page()
        self.init_plottings_page()
        self.window.show_all()

    def __init__(self, filename, urange, hrange):
        """Initialization of the window.
            * funcitons: List of strings representing stuff to be plotted
            * auxiliaries: List of strings of the form "varname=varcomputationstuff" ("intermediate vars")
        """
        # Read arguments
        self.filename = filename
        self.urange = urange
        self.hrange = hrange
        # Default plotting settings
        self.default_samples = 100
        self.default_isosamples = 10
        self.default_h_lower, self.default_h_upper = self.hrange
        self.default_u_lower, self.default_u_upper = self.urange
        self.default_intstops = 5
        self.default_h = (hrange[0]+hrange[1])/2
        self.default_u = (urange[0]+urange[1])/2

        # Initialize plotters
        self.plots = Plotter(filename=filename, integrate=True, intstops=self.default_intstops)
        self.plots.gp('set style line 1 linecolor rgb "black"')
        self.plots.gp('set xrange [%d:%d]' % hrange)
        self.plots.gp('set yrange [%d:%d]' % urange)
        self.variables = self.plots.getvars(sort=True)

        # Set default plotting variable
        self.plotvars = (self.variables[0], self.variables[0])

        # Initialize Gtk layout
        self.init_gtk_layout()

        # first round!
        self.plot()

    def main(self):
        """Just fires the gtk main loop."""
        gtk.main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urange", default=(-5,5))
    parser.add_argument("--hrange", default=(8,12))
    parser.add_argument("-f", "--file", default="wot")
    args = parser.parse_args()
    simpleplotter = SimplePlotter(filename=args.file, urange=args.urange, hrange=args.hrange)
    simpleplotter.main()


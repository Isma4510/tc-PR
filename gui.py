import tkinter as tk
from tkinter import ttk, messagebox
import json
import numpy as np
import sys
import os
from scipy.optimize import brentq

# Matplotlib integration for Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import the model
# Ensure tc_pr_eos.py is in the same directory
try:
    from tc_pr_eos import TC_PR_EOS, R
except ImportError:
    # Fallback if run from a parent directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tc-PR'))
    try:
        from tc_pr_eos import TC_PR_EOS, R
    except ImportError:
        messagebox.showerror("Critical Error", "Could not find 'tc_pr_eos.py'.\nPlease make sure the file is present.")
        sys.exit(1)

# Configuration
JSON_FILE = 'tc_pr_substances.json'

class ThermoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("tc-PR Thermodynamics Calculator")
        self.root.geometry("1000x750")
        
        # Data & Model
        self.substances_db = {}
        self.all_fluid_names = [] # For search filtering
        self.model = None
        self.fluid_name = None
        
        # Load Data
        self.load_database()
        
        # UI Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # --- GUI Layout ---
        self.create_header()
        self.create_notebook()

    def load_database(self):
        """Loads the JSON database."""
        # Look in current dir or subfolder
        paths = [JSON_FILE, os.path.join('tc-PR', JSON_FILE)]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.substances_db = json.load(f)
                        self.all_fluid_names = sorted(list(self.substances_db.keys()))
                    return
                except Exception as e:
                    messagebox.showerror("JSON Error", f"Error reading database:\n{e}")
                    return
        
        messagebox.showerror("File Error", f"Could not find '{JSON_FILE}'")

    def create_header(self):
        """Top section with Fluid Search and Selection."""
        header_frame = ttk.LabelFrame(self.root, text="Fluid Configuration", padding=10)
        header_frame.pack(fill="x", padx=10, pady=5)

        # --- Search & Select Container ---
        sel_frame = ttk.Frame(header_frame)
        sel_frame.pack(fill="x")

        # 1. Search Bar
        ttk.Label(sel_frame, text="Search:", font=('Helvetica', 10)).pack(side="left", padx=(5, 2))
        self.search_var = tk.StringVar()
        self.entry_search = ttk.Entry(sel_frame, textvariable=self.search_var, width=15)
        self.entry_search.pack(side="left", padx=5)
        self.entry_search.bind('<KeyRelease>', self.filter_fluids)

        # 2. Dropdown
        ttk.Label(sel_frame, text="Select Fluid:", font=('Helvetica', 10, 'bold')).pack(side="left", padx=(15, 5))
        self.combo_fluid = ttk.Combobox(sel_frame, values=self.all_fluid_names, state="readonly", width=25)
        self.combo_fluid.pack(side="left", padx=5)
        self.combo_fluid.bind("<<ComboboxSelected>>", self.on_fluid_change)
        
        # 3. Info Display
        self.lbl_info = ttk.Label(sel_frame, text="No fluid selected", foreground="gray")
        self.lbl_info.pack(side="left", padx=20)

    def filter_fluids(self, event):
        """Filters the combobox values based on search text."""
        search_term = self.search_var.get().lower()
        if search_term == "":
            self.combo_fluid['values'] = self.all_fluid_names
        else:
            filtered_list = [name for name in self.all_fluid_names if search_term in name.lower()]
            self.combo_fluid['values'] = filtered_list
        
        # Optional: Auto-select if only one match
        # if len(self.combo_fluid['values']) == 1:
        #     self.combo_fluid.current(0)
        #     self.on_fluid_change(None)

    def on_fluid_change(self, event):
        """Initializes the EOS model when a fluid is picked."""
        name = self.combo_fluid.get()
        if name in self.substances_db:
            self.fluid_name = name
            params = self.substances_db[name]
            try:
                self.model = TC_PR_EOS(params)
                # Update info label
                self.lbl_info.config(
                    text=f"Tc = {self.model.Tc:.2f} K  |  Pc = {self.model.Pc/1e5:.2f} bar  |  ω = {self.model.omega:.4f}",
                    foreground="black"
                )
                # Reset results and plots
                self.ax.clear()
                self.canvas.draw()
                self.txt_result.delete("1.0", tk.END)
            except Exception as e:
                messagebox.showerror("Model Error", f"Error initializing fluid '{name}':\n{e}")

    def create_notebook(self):
        """Creates the tabs."""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Tab 1: Property Calculator
        self.tab_calc = ttk.Frame(notebook)
        notebook.add(self.tab_calc, text="Property Calculator")
        self.setup_calculator_tab()

        # Tab 2: Saturation Plot
        self.tab_plot = ttk.Frame(notebook)
        notebook.add(self.tab_plot, text="Saturation Curve")
        self.setup_plot_tab()

    # =========================================================================
    # TAB 1: FLEXIBLE CALCULATOR
    # =========================================================================
    def setup_calculator_tab(self):
        """Setup for the calculation tab."""
        main_frame = ttk.Frame(self.tab_calc, padding=15)
        main_frame.pack(fill="both", expand=True)

        # -- Mode Selection --
        mode_frame = ttk.LabelFrame(main_frame, text="Calculation Mode (Inputs -> Outputs)", padding=10)
        mode_frame.pack(fill="x", pady=5)

        self.calc_mode = tk.StringVar(value="TP")
        
        modes = [
            ("Temperature (T) & Pressure (P)  ->  Molar Volume (V), Z", "TP"),
            ("Temperature (T) & Volume (V)    ->  Pressure (P), Z", "TV"),
            ("Pressure (P) & Volume (V)       ->  Temperature (T), Z", "PV")
        ]

        for text, mode in modes:
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.calc_mode, value=mode, command=self.update_inputs)
            rb.pack(anchor="w", pady=2)

        # -- Input Fields (Dynamic) --
        input_frame = ttk.LabelFrame(main_frame, text="Input Values", padding=10)
        input_frame.pack(fill="x", pady=10)

        # Container frames for alignment
        self.frame_input_1 = ttk.Frame(input_frame)
        self.frame_input_1.pack(fill="x", pady=5)
        self.frame_input_2 = ttk.Frame(input_frame)
        self.frame_input_2.pack(fill="x", pady=5)

        # Input 1
        self.lbl_in1 = ttk.Label(self.frame_input_1, text="Input 1", width=20)
        self.lbl_in1.pack(side="left")
        self.ent_in1 = ttk.Entry(self.frame_input_1)
        self.ent_in1.pack(side="left", fill="x", expand=True)
        self.lbl_unit1 = ttk.Label(self.frame_input_1, text="unit", width=10)
        self.lbl_unit1.pack(side="left", padx=5)

        # Input 2
        self.lbl_in2 = ttk.Label(self.frame_input_2, text="Input 2", width=20)
        self.lbl_in2.pack(side="left")
        self.ent_in2 = ttk.Entry(self.frame_input_2)
        self.ent_in2.pack(side="left", fill="x", expand=True)
        self.lbl_unit2 = ttk.Label(self.frame_input_2, text="unit", width=10)
        self.lbl_unit2.pack(side="left", padx=5)

        # Calculate Button
        btn_calc = ttk.Button(main_frame, text="CALCULATE", command=self.perform_calculation)
        btn_calc.pack(pady=10, ipadx=20, ipady=5)

        # -- Results Area --
        res_frame = ttk.LabelFrame(main_frame, text="Detailed Results", padding=10)
        res_frame.pack(fill="both", expand=True)
        
        self.txt_result = tk.Text(res_frame, height=10, font=("Consolas", 10))
        self.txt_result.pack(fill="both", expand=True)

        # Initialize fields
        self.update_inputs()

    def update_inputs(self):
        """Updates labels and units based on selected mode."""
        mode = self.calc_mode.get()
        
        # Clear entries
        self.ent_in1.delete(0, tk.END)
        self.ent_in2.delete(0, tk.END)

        if mode == "TP":
            self.lbl_in1.config(text="Temperature (T):")
            self.lbl_unit1.config(text="K")
            self.ent_in1.insert(0, "300")
            
            self.lbl_in2.config(text="Pressure (P):")
            self.lbl_unit2.config(text="bar")
            self.ent_in2.insert(0, "10")
            
        elif mode == "TV":
            self.lbl_in1.config(text="Temperature (T):")
            self.lbl_unit1.config(text="K")
            self.ent_in1.insert(0, "300")
            
            self.lbl_in2.config(text="Molar Volume (Vm):")
            self.lbl_unit2.config(text="m³/mol")
            self.ent_in2.insert(0, "0.001")

        elif mode == "PV":
            self.lbl_in1.config(text="Pressure (P):")
            self.lbl_unit1.config(text="bar")
            self.ent_in1.insert(0, "10")
            
            self.lbl_in2.config(text="Molar Volume (Vm):")
            self.lbl_unit2.config(text="m³/mol")
            self.ent_in2.insert(0, "0.001")

    def perform_calculation(self):
        """Executes the thermodynamic calculation."""
        if not self.model:
            messagebox.showwarning("Warning", "Please select a fluid first.")
            return

        mode = self.calc_mode.get()
        self.txt_result.delete("1.0", tk.END)

        try:
            v1 = float(self.ent_in1.get())
            v2 = float(self.ent_in2.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
            return

        results = ""

        try:
            # --- CASE 1: Given T, P -> Find V, Z ---
            if mode == "TP":
                T_K = v1
                P_Pa = v2 * 1e5 # Convert bar to Pa
                
                if T_K <= 0 or P_Pa <= 0: raise ValueError("T and P must be positive.")

                # Solve cubic EOS for Z
                roots = self.model.solve_eos_for_z(T_K, P_Pa)
                
                results += f"--- Inputs ---\nTemperature : {T_K:.2f} K\nPressure    : {v2:.4f} bar ({P_Pa:.2e} Pa)\n\n"
                results += f"--- Outputs ---\n"
                
                if not roots:
                    results += "No valid Z roots found (Unphysical condition?)\n"
                else:
                    # Identify phases based on number of roots
                    labels = ["Liquid", "Vapor"] if len(roots) > 1 else ["Single Phase"]
                    
                    for i, Z in enumerate(roots):
                        Vm = Z * R * T_K / P_Pa
                        rho = 1.0 / Vm
                        # If we have more roots than labels (rare), fallback to generic index
                        label = labels[i] if i < len(labels) else f"Root {i+1}"
                        
                        results += f"[{label}]\n"
                        results += f"  Z Factor        : {Z:.5f}\n"
                        results += f"  Molar Volume    : {Vm:.6e} m³/mol\n"
                        results += f"  Molar Density   : {rho:.4f} mol/m³\n\n"

            # --- CASE 2: Given T, V -> Find P, Z ---
            elif mode == "TV":
                T_K = v1
                Vm = v2
                
                if T_K <= 0 or Vm <= 0: raise ValueError("T and Vm must be positive.")
                
                # Direct calculation
                P_Pa = self.model.tc_pr(T_K, Vm)
                P_bar = P_Pa / 1e5
                Z = P_Pa * Vm / (R * T_K)
                
                results += f"--- Inputs ---\nTemperature   : {T_K:.2f} K\nMolar Volume  : {Vm:.6e} m³/mol\n\n"
                results += f"--- Outputs ---\n"
                results += f"Pressure      : {P_bar:.4f} bar\n"
                results += f"                {P_Pa:.2e} Pa\n"
                results += f"Z Factor      : {Z:.5f}\n"

            # --- CASE 3: Given P, V -> Find T, Z ---
            elif mode == "PV":
                P_Pa = v1 * 1e5
                Vm = v2
                
                if P_Pa <= 0 or Vm <= 0: raise ValueError("P and Vm must be positive.")
                
                # We need to solve P_calc(T) - P_target = 0 for T
                # Initial guess: Ideal Gas Law T = PV / R
                T_guess = P_Pa * Vm / R
                
                def objective_func_T(T_var):
                    if T_var <= 0: return -1e9 # Penalty
                    return self.model.tc_pr(T_var, Vm) - P_Pa
                
                # Bracket the root around the guess
                T_min, T_max = T_guess * 0.1, T_guess * 5.0
                
                try:
                    # brentq is robust if we bracket the root
                    T_sol = brentq(objective_func_T, T_min, T_max)
                    Z = P_Pa * Vm / (R * T_sol)
                    
                    results += f"--- Inputs ---\nPressure      : {v1:.4f} bar\nMolar Volume  : {Vm:.6e} m³/mol\n\n"
                    results += f"--- Outputs ---\n"
                    results += f"Temperature   : {T_sol:.2f} K\n"
                    results += f"Z Factor      : {Z:.5f}\n"
                except ValueError:
                    results += "Calculation Failed: Could not find a temperature for this (P, V) pair.\n"
                    results += "Check if the volume is physically possible at this pressure.\n"

        except Exception as e:
            results = f"Calculation Error: {str(e)}"

        self.txt_result.insert(tk.END, results)


    # =========================================================================
    # TAB 2: SATURATION PLOT
    # =========================================================================
    def setup_plot_tab(self):
        """Setup for the plotting tab."""
        plot_frame = ttk.Frame(self.tab_plot)
        plot_frame.pack(fill="both", expand=True)

        # Toolbar Frame
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x")
        
        btn_plot = ttk.Button(toolbar_frame, text="Plot Saturation Curve", command=self.plot_psat)
        btn_plot.pack(side="left", padx=10, pady=5)
        
        lbl_hint = ttk.Label(toolbar_frame, text="(Range: 0.4*Tc to 0.99*Tc)", font=("Helvetica", 9, "italic"))
        lbl_hint.pack(side="left")

        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Matplotlib Toolbar (Zoom, Save, etc.)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def plot_psat(self):
        """Generates the Psat vs T plot."""
        if not self.model:
            return

        Tc = self.model.Tc
        # Define range (avoid 0 and exactly Tc to prevent singularity)
        T_range = np.linspace(0.4 * Tc, 0.99 * Tc, 100)
        P_vals = []
        T_valid = []

        # Visual feedback
        self.root.config(cursor="wait")
        self.root.update()

        for T in T_range:
            try:
                # Calculate Psat in Pa
                p = self.model.calculate_psat(T)
                if p is not None:
                    T_valid.append(T)
                    P_vals.append(p / 1e5) # Convert to bar
            except:
                pass
        
        self.root.config(cursor="")

        # Plotting
        self.ax.clear()
        if T_valid:
            self.ax.plot(T_valid, P_vals, color='#E63946', linewidth=2, label='P_sat (tc-PR)')
            self.ax.set_title(f"Saturation Curve: {self.fluid_name}", fontsize=12, fontweight='bold')
            self.ax.set_xlabel("Temperature (K)")
            self.ax.set_ylabel("Pressure (bar)")
            self.ax.grid(True, linestyle=':', alpha=0.7)
            self.ax.legend()
            
            # Mark Critical Point
            self.ax.plot(Tc, self.model.Pc/1e5, 'k*', markersize=12, label='Critical Point')
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, "Could not calculate curve.\n(Fluid might be supercritical or parameters invalid)", 
                         ha='center', va='center', transform=self.ax.transAxes)
        
        self.canvas.draw()

# --- Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ThermoApp(root)
    root.mainloop()
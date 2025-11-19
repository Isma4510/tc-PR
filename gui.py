import tkinter as tk
from tkinter import ttk, messagebox
import json
import numpy as np
import sys
import os
from scipy.optimize import brentq

# Matplotlib integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import model
try:
    from tc_pr_eos import TC_PR_EOS, R
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tc-PR'))
    try:
        from tc_pr_eos import TC_PR_EOS, R
    except ImportError:
        messagebox.showerror("Critical Error", "Could not find 'tc_pr_eos.py'.")
        sys.exit(1)

JSON_FILE = 'tc_pr_substances.json'

class ThermoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced tc-PR Thermodynamics Calculator")
        self.root.geometry("1100x800")
        
        self.substances_db = {}
        self.all_fluid_names = []
        self.model = None
        self.fluid_name = None
        
        self.load_database()
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.create_header()
        self.create_notebook()

    def load_database(self):
        """Loads the JSON database."""
        paths = [JSON_FILE, os.path.join('tc-PR', JSON_FILE)]
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.substances_db = json.load(f)
                    self.all_fluid_names = sorted(list(self.substances_db.keys()))
                return
        messagebox.showerror("File Error", f"Could not find '{JSON_FILE}'")

    def create_header(self):
        """Fluid Selection Header"""
        header_frame = ttk.LabelFrame(self.root, text="Fluid Selection", padding=10)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        frame_top = ttk.Frame(header_frame)
        frame_top.pack(fill="x")

        ttk.Label(frame_top, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        self.entry_search = ttk.Entry(frame_top, textvariable=self.search_var, width=15)
        self.entry_search.pack(side="left", padx=5)
        self.entry_search.bind('<KeyRelease>', self.filter_fluids)

        ttk.Label(frame_top, text="Fluid:").pack(side="left", padx=10)
        self.combo_fluid = ttk.Combobox(frame_top, values=self.all_fluid_names, state="readonly", width=25)
        self.combo_fluid.pack(side="left")
        self.combo_fluid.bind("<<ComboboxSelected>>", self.on_fluid_change)
        
        self.lbl_info = ttk.Label(frame_top, text="No fluid selected", foreground="gray")
        self.lbl_info.pack(side="left", padx=20)

    def filter_fluids(self, event):
        """Filters dropdown list."""
        term = self.search_var.get().lower()
        filtered = [n for n in self.all_fluid_names if term in n.lower()]
        self.combo_fluid['values'] = filtered

    def on_fluid_change(self, event):
        """Initializes model and autofills Cp params if available."""
        name = self.combo_fluid.get()
        if name in self.substances_db:
            self.fluid_name = name
            params = self.substances_db[name]
            try:
                self.model = TC_PR_EOS(params)
                self.lbl_info.config(
                    text=f"Tc={self.model.Tc:.2f}K  Pc={self.model.Pc/1e5:.2f}bar  ω={self.model.omega:.4f}",
                    foreground="black"
                )
                # Auto-fill Cp parameters if they exist in JSON
                self.populate_cp_fields(params)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def populate_cp_fields(self, params):
        """Fills Cp input fields from JSON data."""
        # Clear fields first
        for entry in self.cp_entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, "0.0")
        # Try to read cp params using helper (handles several JSON key names)
        cp_from_json = None
        try:
            cp_from_json = TC_PR_EOS.get_cp_params_from_json(self.fluid_name, json_path=JSON_FILE)
        except Exception:
            cp_from_json = None

        if cp_from_json:
            for key in ["A", "B", "C", "D", "E"]:
                if key in cp_from_json:
                    self.cp_entries[key].delete(0, tk.END)
                    self.cp_entries[key].insert(0, str(cp_from_json[key]))
            return

        # Fallback to old key if present
        if "cp_ideal_coeffs" in params:
            coeffs = params["cp_ideal_coeffs"]
            for key in ["A", "B", "C", "D", "E"]:
                if key in coeffs:
                    self.cp_entries[key].delete(0, tk.END)
                    self.cp_entries[key].insert(0, str(coeffs[key]))

    def create_notebook(self):
        """Creates tabs."""
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.tab_calc = ttk.Frame(nb)
        nb.add(self.tab_calc, text="Properties Calculator")
        self.setup_calculator_tab()
        
        self.tab_plot = ttk.Frame(nb)
        nb.add(self.tab_plot, text="Saturation Curve")
        self.setup_plot_tab()

    # =========================================================================
    # TAB 1: CALCULATOR
    # =========================================================================
    def setup_calculator_tab(self):
        main_frame = ttk.Frame(self.tab_calc)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Left Column: Inputs ---
        left_col = ttk.Frame(main_frame)
        left_col.pack(side="left", fill="y", padx=5)

        # 1. Calculation Mode
        mode_frame = ttk.LabelFrame(left_col, text="Calculation Mode", padding=10)
        mode_frame.pack(fill="x", pady=5)
        
        # Use a combobox (list) to choose input property pair
        self.calc_mode = tk.StringVar(value="TP")
        modes = [
            ("T (K) & P (bar)", "TP"),
            ("T (K) & V (m³/mol)", "TV"),
            ("P (bar) & V (m³/mol)", "PV"),
            ("H (J/mol) & S (J/mol.K)", "HS"),
            ("P (bar) & H (J/mol)", "PH"),
            ("T (K) & S (J/mol.K)", "TS")
        ]
        self.mode_map = {label: code for label, code in modes}
        self.mode_display = [label for label, code in modes]
        self.combo_mode = ttk.Combobox(mode_frame, values=self.mode_display, state='readonly')
        self.combo_mode.set(self.mode_display[0])
        self.combo_mode.pack(fill='x')
        self.combo_mode.bind('<<ComboboxSelected>>', lambda e: self.update_inputs())

        # 2. State Inputs
        self.input_frame = ttk.LabelFrame(left_col, text="State Variables", padding=10)
        self.input_frame.pack(fill="x", pady=5)
        
        self.lbl_1 = ttk.Label(self.input_frame, text="Input 1")
        self.lbl_1.grid(row=0, column=0, sticky="w")
        self.ent_1 = ttk.Entry(self.input_frame, width=12)
        self.ent_1.grid(row=0, column=1, padx=5)
        
        self.lbl_2 = ttk.Label(self.input_frame, text="Input 2")
        self.lbl_2.grid(row=1, column=0, sticky="w")
        self.ent_2 = ttk.Entry(self.input_frame, width=12)
        self.ent_2.grid(row=1, column=1, padx=5)

        # 3. Cp Parameters Input
        cp_frame = ttk.LabelFrame(left_col, text="Ideal Gas Cp Parameters (J/mol.K)", padding=10)
        cp_frame.pack(fill="x", pady=5)
        ttk.Label(cp_frame, text="Cp = A + BT + CT² + DT³ + ET⁴").grid(row=0, column=0, columnspan=2, pady=(0,5))
        
        self.cp_entries = {}
        for i, label in enumerate(["A", "B", "C", "D", "E"]):
            ttk.Label(cp_frame, text=f"{label}:").grid(row=i+1, column=0, sticky="e")
            ent = ttk.Entry(cp_frame, width=15)
            ent.insert(0, "0.0")
            ent.grid(row=i+1, column=1, padx=5, pady=2)
            self.cp_entries[label] = ent

        # Button to save Cp params back to JSON
        ttk.Button(cp_frame, text="Save Cp to JSON", command=self.save_cp_to_json).grid(row=6, column=0, columnspan=2, pady=(6,2))

        ttk.Button(left_col, text="CALCULATE", command=self.calculate).pack(fill="x", pady=15)

        # --- Right Column: Results ---
        right_col = ttk.LabelFrame(main_frame, text="Results (H, S relative to Ideal Gas at 298.15K)", padding=10)
        right_col.pack(side="left", fill="both", expand=True, padx=5)
        
        self.txt_res = tk.Text(right_col, font=("Consolas", 10), width=60)
        self.txt_res.pack(fill="both", expand=True)
        
        self.update_inputs()

    def update_inputs(self):
        """Updates input labels based on mode."""
        # determine selected code from combobox or var
        sel = None
        try:
            sel = self.combo_mode.get()
        except Exception:
            sel = None

        if sel and hasattr(self, 'mode_map'):
            m = self.mode_map.get(sel, 'TP')
        else:
            m = self.calc_mode.get()
        self.ent_1.delete(0, tk.END)
        self.ent_2.delete(0, tk.END)
        
        if m == "TP":
            self.lbl_1.config(text="Temp (K):")
            self.ent_1.insert(0, "300")
            self.lbl_2.config(text="Pres (bar):")
            self.ent_2.insert(0, "1")
        elif m == "TV":
            self.lbl_1.config(text="Temp (K):")
            self.ent_1.insert(0, "300")
            self.lbl_2.config(text="Vol (m³/mol):")
            self.ent_2.insert(0, "0.024")
        elif m == "PV":
            self.lbl_1.config(text="Pres (bar):")
            self.ent_1.insert(0, "1")
            self.lbl_2.config(text="Vol (m³/mol):")
            self.ent_2.insert(0, "0.024")
        elif m == "HS":
            self.lbl_1.config(text="Enthalpy H (J/mol):")
            self.ent_1.insert(0, "0.0")
            self.lbl_2.config(text="Entropy S (J/mol.K):")
            self.ent_2.insert(0, "0.0")
        elif m == "PH":
            self.lbl_1.config(text="Pres (bar):")
            self.ent_1.insert(0, "1")
            self.lbl_2.config(text="Enthalpy H (J/mol):")
            self.ent_2.insert(0, "0.0")
        elif m == "TS":
            self.lbl_1.config(text="Temp (K):")
            self.ent_1.insert(0, "300")
            self.lbl_2.config(text="Entropy S (J/mol.K):")
            self.ent_2.insert(0, "0.0")

    def get_cp_params(self):
        """Extracts Cp params from GUI."""
        params = {}
        try:
            for k, ent in self.cp_entries.items():
                params[k] = float(ent.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid Cp parameters (must be numbers).")
            return None
        return params

    def calculate(self):
        """Main calculation logic."""
        if not self.model:
            messagebox.showwarning("Warning", "Select a fluid first.")
            return
        
        cp_params = self.get_cp_params()
        if cp_params is None: return
        
        self.txt_res.delete("1.0", tk.END)
        mode = self.calc_mode.get()
        
        try:
            v1 = float(self.ent_1.get())
            v2 = float(self.ent_2.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric inputs for state variables.")
            return

        try:
            solutions = [] # List of dicts {'phase', 'T', 'P', 'V', 'Z'}

            # --- Mode TP: T, P -> V ---
            if mode == "TP":
                T, P = v1, v2 * 1e5 # bar -> Pa
                if T<=0 or P<=0: raise ValueError("T and P must be > 0")
                
                roots = self.model.solve_eos_for_z(T, P)
                if not roots:
                    self.txt_res.insert(tk.END, "No valid roots found.\n"); return

                # If three roots, the middle root is non-physical in this UI: show only liquid and vapor
                if len(roots) >= 2:
                    Zs = [min(roots), max(roots)]
                    labels = ["Liquid", "Vapor"]
                else:
                    Zs = roots
                    labels = ["Single Phase"]

                for i, Z in enumerate(Zs):
                    V = Z * R * T / P
                    lbl = labels[i] if i < len(labels) else f"Root {i}"
                    solutions.append({'phase': lbl, 'T': T, 'P': P, 'V': V, 'Z': Z})

            # --- Mode TV: T, V -> P ---
            elif mode == "TV":
                T, V = v1, v2
                if T<=0 or V<=0: raise ValueError("T and V must be > 0")
                
                P = self.model.tc_pr(T, V)
                Z = P * V / (R * T)
                solutions.append({'phase': "Single Phase", 'T': T, 'P': P, 'V': V, 'Z': Z})

            # --- Mode PV: P, V -> T ---
            elif mode == "PV":
                P, V = v1 * 1e5, v2
                if P<=0 or V<=0: raise ValueError("P and V must be > 0")
                
                # Initial guess T = PV/R
                T_guess = P * V / R
                def obj(t): return self.model.tc_pr(t, V) - P if t > 0 else 1e9
                
                try:
                    # Bracket the root
                    T = brentq(obj, T_guess*0.1, T_guess*5.0)
                    Z = P * V / (R * T)
                    solutions.append({'phase': "Single Phase", 'T': T, 'P': P, 'V': V, 'Z': Z})
                except:
                    self.txt_res.insert(tk.END, "Could not find T for given P, V.\n"); return

            # --- Mode HS: H, S -> T, P, V ---
            elif mode == "HS":
                H_target, S_target = v1, v2
                # try to solve for T and P
                res = self.model.solve_from_HS(H_target, S_target, cp_params, phase='auto')
                if not res.get('success'):
                    self.txt_res.insert(tk.END, f"Could not invert H/S: {res.get('message')}\n")
                    return
                T = res['T']
                P = res['P']
                Z = res['Z']
                V = res['v']
                solutions.append({'phase': "HS Solution", 'T': T, 'P': P, 'V': V, 'Z': Z})

            # --- Mode PH: P, H -> T, V ---
            elif mode == "PH":
                P_bar, H_target = v1, v2
                P = P_bar * 1e5
                res = self.model.solve_from_PH(P, H_target, cp_params, phase='auto')
                if not res.get('success'):
                    self.txt_res.insert(tk.END, f"Could not invert P/H: {res.get('message')}\n")
                    return
                T = res['T']
                P = res['P']
                Z = res['Z']
                V = res['v']
                solutions.append({'phase': "PH Solution", 'T': T, 'P': P, 'V': V, 'Z': Z})

            # --- Mode TS: T, S -> P, V ---
            elif mode == "TS":
                T, S_target = v1, v2
                res = self.model.solve_from_TS(T, S_target, cp_params, phase='auto')
                if not res.get('success'):
                    self.txt_res.insert(tk.END, f"Could not invert T/S: {res.get('message')}\n")
                    return
                T = res['T']
                P = res['P']
                Z = res['Z']
                V = res['v']
                solutions.append({'phase': "TS Solution", 'T': T, 'P': P, 'V': V, 'Z': Z})

            # If biphasic at solution conditions, compute explicit liquid/vapor entropies
            sat = None
            if solutions:
                T0 = solutions[0]['T']
                P0 = solutions[0]['P']
                try:
                    z_roots = self.model.solve_eos_for_z(T0, P0)
                    if z_roots and len(z_roots) > 1:
                        Z_L, Z_V = min(z_roots), max(z_roots)
                        h_l, s_l = self.model.calculate_properties(T0, P0, Z_L, cp_params)
                        h_v, s_v = self.model.calculate_properties(T0, P0, Z_V, cp_params)
                        sat = {'Psat': P0, 's_liq': s_l, 's_vap': s_v, 'h_liq': h_l, 'h_vap': h_v}
                except Exception:
                    sat = None

            # Calculate H and S for each solution and display
            out = ""
            for sol in solutions:
                T, P, Z, V = sol['T'], sol['P'], sol['Z'], sol['V']
                H, S = self.model.calculate_properties(T, P, Z, cp_params)
                
                out += f"=== {sol['phase']} ===\n"
                out += f"  Temperature : {T:.2f} K\n"
                out += f"  Pressure    : {P/1e5:.4f} bar ({P:.2e} Pa)\n"
                out += f"  Molar Vol.  : {V:.6e} m³/mol\n"
                out += f"  Density     : {1/V:.2f} mol/m³\n"
                out += f"  Z Factor    : {Z:.5f}\n"
                out += f"  Enthalpy (H): {H:.2f} J/mol\n"
                out += f"  Entropy (S) : {S:.2f} J/mol.K\n\n"
            
            if sat is not None:
                out += "--- Saturation entropies (liquid / vapor) ---\n"
                out += f"  Psat        : {sat['Psat']/1e5:.6f} bar\n"
                out += f"  s_liq       : {sat['s_liq']:.4f} J/mol.K\n"
                out += f"  s_vap       : {sat['s_vap']:.4f} J/mol.K\n"
                out += f"  h_liq       : {sat['h_liq']:.2f} J/mol\n"
                out += f"  h_vap       : {sat['h_vap']:.2f} J/mol\n\n"

            self.txt_res.insert(tk.END, out)

        except Exception as e:
            self.txt_res.insert(tk.END, f"Calculation Error:\n{e}")

    def save_cp_to_json(self):
        """Save current Cp entries into the JSON database using TC_PR_EOS helper."""
        if not self.fluid_name:
            messagebox.showwarning("Warning", "Select a fluid first.")
            return
        cp = self.get_cp_params()
        if cp is None:
            return
        try:
            TC_PR_EOS.save_cp_params_to_json(self.fluid_name, cp, json_path=JSON_FILE)
            messagebox.showinfo("Saved", f"Cp parameters saved to {JSON_FILE} for {self.fluid_name}.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save Cp params: {e}")

    # =========================================================================
    # TAB 2: PLOTTING
    # =========================================================================
    def setup_plot_tab(self):
        frame = ttk.Frame(self.tab_plot)
        frame.pack(fill="both", expand=True)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Plot Saturation Curve", command=self.plot).pack(pady=5)
        
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).pack()

    def plot(self):
        if not self.model: return
        Tc = self.model.Tc
        T_vals = np.linspace(0.45*Tc, 0.99*Tc, 80)
        P_vals = []
        valid_T = []
        
        self.root.config(cursor="wait")
        self.root.update()
        
        for T in T_vals:
            try:
                p = self.model.calculate_psat(T)
                if p:
                    valid_T.append(T)
                    P_vals.append(p/1e5)
            except: pass
        
        self.root.config(cursor="")
        
        self.ax.clear()
        self.ax.plot(valid_T, P_vals, 'b-', lw=2)
        self.ax.set_xlabel("Temperature (K)")
        self.ax.set_ylabel("Pressure (bar)")
        self.ax.set_title(f"Saturation Pressure: {self.fluid_name}")
        self.ax.grid(True, ls="--")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermoApp(root)
    root.mainloop()
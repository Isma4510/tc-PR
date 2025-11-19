import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# Import CoolProp
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("Error: CoolProp library is not installed. Run 'pip install CoolProp'")
    sys.exit(1)

# Import your tc-PR class
try:
    from tc_pr_eos import TC_PR_EOS
except ImportError:
    print("Error: Could not import TC_PR_EOS from 'tc_pr_eos.py'")
    sys.exit(1)

# --- Configuration ---
JSON_FILE = 'tc_pr_substances.json'

# Extended Dictionary: JSON Key -> CoolProp Name
FLUIDS_TO_TEST = {
    # --- Classical Gases ---
    "CARBON DIOXIDE": "CO2",
    "NITROGEN": "Nitrogen",
    "OXYGEN": "Oxygen",
    "ARGON": "Argon",
    "WATER": "Water",
    "HELIUM-4": "Helium",
    
    # --- Hydrocarbons (Natural Refrigerants) ---
    "METHANE": "Methane",     # R50
    "ETHANE": "Ethane",       # R170
    "PROPANE": "n-Propane",   # R290
    "n-BUTANE": "n-Butane",   # R600
    "ISOBUTANE": "IsoButane", # R600a

    # --- Synthetic Refrigerants (HFC / HCFC) ---
    "1,1,1,2-TETRAFLUOROETHANE": "R134a", # Very common (auto AC)
    "CHLORODIFLUOROMETHANE": "R22",       # Older standard
    "DIFLUOROMETHANE": "R32",             # New standard
    "PENTAFLUOROETHANE": "R125",          # Component of R410A
    
    # --- Other Technical Fluids ---
    "AMMONIA": "Ammonia",     # R717 (Industrial)
    "SULFUR HEXAFLUORIDE": "SF6"
}

def load_substances(filename):
    """Loads the substances database from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)

def calculate_metrics(fluid_name, cp_name, params):
    """
    Compares the tc-PR model with CoolProp for a given fluid.
    Returns: AAD (%), Valid Temperatures, Deviations
    """
    try:
        model = TC_PR_EOS(params)
    except Exception as e:
        # Some fluids might have issues in initialization
        # print(f"  [Info] Init error for {fluid_name}: {e}")
        return None, None, None

    Tc = model.Tc
    
    # Temperature range: 0.55*Tc to 0.99*Tc
    # Use PropsSI to get the real triple point if possible
    try:
        T_triple = PropsSI('Ttriple', cp_name)
    except:
        T_triple = 0.5 * Tc 

    T_min = max(T_triple + 2.0, 0.55 * Tc)
    T_max = 0.99 * Tc
    
    # 40 points per fluid is enough for a good metric
    T_range = np.linspace(T_min, T_max, 40)
    
    P_model_list = []
    P_ref_list = []
    T_valid = []
    
    for T in T_range:
        # 1. tc-PR Model
        P_calc = model.calculate_psat(T)
        
        # 2. CoolProp Reference
        try:
            P_ref = PropsSI('P', 'T', T, 'Q', 1, cp_name)
        except:
            P_ref = None
            
        if P_calc is not None and P_ref is not None:
            P_model_list.append(P_calc)
            P_ref_list.append(P_ref)
            T_valid.append(T)
            
    if not P_model_list:
        return None, None, None

    P_model = np.array(P_model_list)
    P_ref = np.array(P_ref_list)
    T_valid = np.array(T_valid)
    
    # Relative Deviation (%)
    deviations = (P_model - P_ref) / P_ref * 100.0
    
    # AAD = Mean Absolute Deviation
    aad = np.mean(np.abs(deviations))
    
    return aad, T_valid, deviations

def main():
    print(f"{'='*60}")
    print(f" Psat COMPARISON: tc-PR vs COOLPROP")
    print(f"{'='*60}")
    
    db = load_substances(JSON_FILE)
    
    results = [] # List to store (Name, AAD, N_pts)
    plot_data = {}

    # Table Header
    print(f"{'FLUID (JSON)':<30} | {'COOLPROP':<10} | {'AAD (%)':<8} | {'PTS'}")
    print("-" * 60)

    for json_key, cp_name in FLUIDS_TO_TEST.items():
        if json_key not in db:
            continue
            
        params = db[json_key]
        aad, T_vals, devs = calculate_metrics(json_key, cp_name, params)
        
        if aad is not None:
            # Clean console output
            # Truncate JSON name if too long
            disp_name = (json_key[:27] + '..') if len(json_key) > 29 else json_key
            print(f"{disp_name:<30} | {cp_name:<10} | {aad:6.3f} % | {len(T_vals)}")
            
            results.append((json_key, aad))
            # Store reduced temperature for comparing different fluids on the same plot
            plot_data[json_key] = (T_vals / params['Tc'], devs, cp_name)
        else:
            print(f"{json_key:<30} | {cp_name:<10} |   FAIL   | 0")

    # --- Generate Plot ---
    plt.figure(figsize=(12, 7))
    
    # Color palette and markers to distinguish groups
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
    
    # Sort to display legends cleanly (e.g., by increasing AAD)
    # This ensures the legend order matches performance or name
    sorted_plots = sorted(plot_data.items(), key=lambda x: x[0])

    for i, (name, (Tr, dev, cp_label)) in enumerate(sorted_plots):
        # Display only the short name (CoolProp) in the legend to avoid clutter
        marker = markers[i % len(markers)]
        
        # Fine dotted line to connect, markers every few points
        plt.plot(Tr, dev, marker=marker, markevery=4, markersize=6, 
                 linewidth=1.5, alpha=0.8, label=f"{cp_label}")

    plt.axhline(0, color='black', linewidth=1.5)
    plt.xlabel("Reduced Temperature ($T/T_c$)", fontsize=12)
    plt.ylabel("Relative Deviation of $P_{sat}$ (%)", fontsize=12)
    plt.title("Accuracy of tc-PR Model for Various Fluids (Gases, HFCs, HCs)", fontsize=14)
    
    # Legend in 2 columns to handle the large number of fluids
    plt.legend(ncol=2, fontsize=9, loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Smart Zoom: Focus on +/- 3% as tc-PR is usually very accurate
    plt.ylim(-3, 3) 
    
    out_file = "comparison_psat_extended.png"
    print(f"\nPlot saved to: {out_file}")
    plt.savefig(out_file, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
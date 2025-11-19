import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import sys

# Import CoolProp
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("Error: CoolProp is not installed. (pip install CoolProp)")
    sys.exit(1)

# Import tc-PR model
try:
    from tc_pr_eos import TC_PR_EOS, R
except ImportError:
    print("Error: Could not find 'tc_pr_eos.py'.")
    sys.exit(1)

# --- Configuration ---
FLUID_NAME = "CO2"          # Name in JSON
CP_FLUID = "CO2"            # Name in CoolProp
JSON_FILE = "tc_pr_substances.json"

# Default parameters for CO2 (Fallback if JSON missing)
CO2_PARAMS_DEFAULT = {
    "name": "Carbon Dioxide",
    "Tc": 304.21, "Pc": 7383000.0, "omega": 0.2236,
    "L": 0.1783, "M": 0.859, "N": 2.4107,
    "c": -1.1368e-06
}

# -----------------------------------------------------------------------------
# 1. Ideal Cp Fitting Function
# -----------------------------------------------------------------------------
def poly_cp(T, A, B, C, D, E):
    """Polynomial for Cp_ideal (J/mol.K)"""
    return A + B*T + C*T**2 + D*T**3 + E*T**4

def fit_ideal_cp(fluid_cp_name, T_min=200, T_max=1000):
    """Retrieves Cp0 data from CoolProp and fits the coefficients."""
    
    # --- CORRECTION: Check temperature bounds ---
    try:
        # Ask CoolProp for the minimum valid temperature for this fluid
        T_limit_low = PropsSI('Tmin', fluid_cp_name)
        
        # If requested T_min is too low, adjust it
        if T_min < T_limit_low:
            print(f"  [Info] T_min adjusted from {T_min} K to {T_limit_low:.2f} K (CoolProp Limit)")
            T_min = T_limit_low + 0.1 # Small safety margin
    except:
        # If error (unknown fluid, etc.), keep default
        pass
    # -----------------------------------------------------------

    print(f"--- 1. Fitting Ideal Cp for {fluid_cp_name} ({T_min:.1f}-{T_max} K) ---")
    
    T_data = np.linspace(T_min, T_max, 100)
    
    # CP0MOLAR: Molar ideal gas heat capacity [J/mol/K]
    # Note: P is arbitrary for ideal gas, but must be within a valid EOS zone
    try:
        Cp_data = np.array([PropsSI('CP0MOLAR', 'T', t, 'P', 101325, fluid_cp_name) for t in T_data])
    except ValueError as e:
        print(f"Critical error calling CoolProp: {e}")
        sys.exit(1)
    
    # Fit
    popt, _ = curve_fit(poly_cp, T_data, Cp_data)
    A, B, C, D, E = popt
    
    # Calculate fit error
    Cp_fit = poly_cp(T_data, *popt)
    aad_fit = np.mean(np.abs((Cp_fit - Cp_data) / Cp_data)) * 100
    
    print(f"  Coefficients found:")
    print(f"  A={A:.4e}, B={B:.4e}, C={C:.4e}, D={D:.4e}, E={E:.4e}")
    print(f"  Mean fit error (AAD) : {aad_fit:.4f} %")
    
    return {"A": A, "B": B, "C": C, "D": D, "E": E}

# -----------------------------------------------------------------------------
# 2. Enthalpy / Entropy Comparison
# -----------------------------------------------------------------------------
def compare_properties(model, cp_params, fluid_cp_name, P_iso_bar=10):
    """Calculates H and S along an isobar and compares with CoolProp."""
    print(f"\n--- 2. H & S Comparison at P = {P_iso_bar} bar ---")
    
    P_iso = P_iso_bar * 1e5
    # T Range (avoid critical point directly to simplify)
    # Wide range: 250K to 500K
    T_range = np.linspace(250, 500, 50)
    
    H_model, S_model = [], []
    H_ref, S_ref = [], []
    T_valid = []
    
    for T in T_range:
        # --- Reference (CoolProp) ---
        try:
            # Hmolar [J/mol], Smolar [J/mol/K]
            h_cp = PropsSI('Hmolar', 'T', T, 'P', P_iso, fluid_cp_name)
            s_cp = PropsSI('Smolar', 'T', T, 'P', P_iso, fluid_cp_name)
        except:
            continue
            
        # --- tc-PR Model ---
        # Solve Z
        z_roots = model.solve_eos_for_z(T, P_iso)
        if not z_roots: continue
        
        # Root selection (Liquid if low T, Vapor if high T/supercritical)
        # Simplification: Take the root yielding density closest to CoolProp 
        # to ensure comparison of the same phase.
        
        rho_cp = PropsSI('Dmolar', 'T', T, 'P', P_iso, fluid_cp_name) # mol/m3
        v_cp = 1.0 / rho_cp
        
        best_Z = None
        min_diff_v = float('inf')
        
        for z in z_roots:
            v_calc = z * R * T / P_iso
            if abs(v_calc - v_cp) < min_diff_v:
                min_diff_v = abs(v_calc - v_cp)
                best_Z = z
        
        if best_Z is None: continue
        
        # Calculate H, S with model
        h_calc, s_calc = model.calculate_properties(T, P_iso, best_Z, cp_params)
        
        T_valid.append(T)
        H_model.append(h_calc)
        S_model.append(s_calc)
        H_ref.append(h_cp)
        S_ref.append(s_cp)

    # --- Reference Alignment (Offset) ---
    # EoS have arbitrary reference states. To compare, align at the first point.
    if not H_model:
        print("No points calculated.")
        return

    offset_H = H_ref[0] - H_model[0]
    offset_S = S_ref[0] - S_model[0]
    
    H_model_shifted = np.array(H_model) + offset_H
    S_model_shifted = np.array(S_model) + offset_S
    H_ref = np.array(H_ref)
    S_ref = np.array(S_ref)
    
    # --- Calculate AAD ---
    # AAD = mean( |(Calc - Ref) / Ref| ) * 100
    # Note: if H crosses 0, AAD can explode. Using standard AAD here.
    
    aad_H = np.mean(np.abs((H_model_shifted - H_ref) / H_ref)) * 100
    aad_S = np.mean(np.abs((S_model_shifted - S_ref) / S_ref)) * 100
    
    print(f"  Applied Offset H : {offset_H:.2e} J/mol")
    print(f"  Applied Offset S : {offset_S:.2e} J/mol.K")
    print(f"  AAD Enthalpy (H) : {aad_H:.4f} %")
    print(f"  AAD Entropy (S)  : {aad_S:.4f} %")
    
    # --- Plot ---
    plt.figure(figsize=(12, 5))
    
    # Enthalpy
    plt.subplot(1, 2, 1)
    plt.plot(T_valid, H_ref/1000, 'k-', lw=2, label='CoolProp')
    plt.plot(T_valid, H_model_shifted/1000, 'r--', lw=2, label='tc-PR (fitted Cp)')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Enthalpy (kJ/mol)")
    plt.title(f"CO2 Enthalpy at {P_iso_bar} bar\n(AAD: {aad_H:.2f}%)")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    # Entropy
    plt.subplot(1, 2, 2)
    plt.plot(T_valid, S_ref, 'k-', lw=2, label='CoolProp')
    plt.plot(T_valid, S_model_shifted, 'b--', lw=2, label='tc-PR (fitted Cp)')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Entropy (J/mol.K)")
    plt.title(f"CO2 Entropy at {P_iso_bar} bar\n(AAD: {aad_S:.2f}%)")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load fluid parameters
    try:
        with open(JSON_FILE, 'r') as f:
            db = json.load(f)
            if FLUID_NAME in db:
                fluid_params = db[FLUID_NAME]
                print(f"Parameters loaded for {FLUID_NAME} from JSON.")
            else:
                print(f"Warning: {FLUID_NAME} not found in JSON. Using defaults.")
                fluid_params = CO2_PARAMS_DEFAULT
    except FileNotFoundError:
        print("Warning: JSON file not found. Using default parameters.")
        fluid_params = CO2_PARAMS_DEFAULT

    # 2. Initialize EoS
    model = TC_PR_EOS(fluid_params)
    
    # 3. Fit Cp
    cp_coeffs = fit_ideal_cp(CP_FLUID)
    
    # 4. Compare and Plot
    compare_properties(model, cp_coeffs, CP_FLUID, P_iso_bar=50)
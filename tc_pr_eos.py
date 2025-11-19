# tc_pr_eos.py

import json
import numpy as np
from scipy.optimize import brentq, root
import warnings

# Universal gas constant (J/(mol*K))
R = 8.31446261815324

# Safety limits for numerical stability
CLIP_EXP_MIN = -700.0
CLIP_EXP_MAX = 700.0
CLIP_LOG_MIN = 1e-300

class TC_PR_EOS:
    """
    Translated-Consistent-Peng-Robinson (tc-PR) Equation of State.
    """
    def __init__(self, substance_params):
        self.params = substance_params
        self.Tc = self.params['Tc']
        self.Pc = self.params['Pc']
        self.omega = self.params.get('omega', 0.0)
        
        # Volume translation (Peneloux)
        if 'c' in self.params and self.params['c'] is not None:
            self.c_si = self.params['c']
        else:
            self.c_si = self._calculate_c_from_omega()

        # Twu91 Alpha function parameters
        self.L = self.params.get('L')
        self.M = self.params.get('M')
        self.N = self.params.get('N')

        # Calculate standard PR parameters at critical point
        self.a_c = 0.45724 * (R**2 * self.Tc**2) / self.Pc
        self.b = 0.07780 * (R * self.Tc) / self.Pc - self.c_si
        
        if self.b <= 0:
            raise ValueError(f"Invalid 'b' parameter (<=0) for {self.params.get('name', 'substance')}.")

        # Parameters for the cubic equation roots
        sqrt_term_inner = (2 * self.b + 4 * self.c_si)**2 - 8 * self.c_si**2 + 4 * self.b**2
        if sqrt_term_inner < 0:
             raise ValueError("Invalid parameters: negative value in sqrt for d/e calculation.")
        
        sqrt_term = np.sqrt(sqrt_term_inner) 
        self.e = ((2 * self.b + 4 * self.c_si) + sqrt_term) / 2
        self.d = ((2 * self.b + 4 * self.c_si) - sqrt_term) / 2

    def _calculate_c_from_omega(self):
        return (R * self.Tc / self.Pc) * (-0.0065 + 0.0198 * self.omega)

    def _calculate_twu91_alpha(self, T):
        if self.L is None or self.M is None or self.N is None:
            raise ValueError("L, M, N parameters missing")
        
        Tr = T / self.Tc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exp_arg = self.L * (1 - Tr**(self.N * self.M))
            exp_term = np.exp(np.clip(exp_arg, CLIP_EXP_MIN, CLIP_EXP_MAX))
            alpha = Tr**(self.N * (self.M - 1)) * exp_term
        return alpha

    def _get_a_param(self, T):
        return self.a_c * self._calculate_twu91_alpha(T)

    def _da_dT(self, T):
        """
        Calculates the temperature derivative of the attractive parameter: da/dT.
        Used for Enthalpy and Entropy calculations.
        """
        if self.L is None: return 0.0
        
        Tr = T / self.Tc
        # Alpha calculation components
        term_exp = np.exp(np.clip(self.L * (1 - Tr**(self.N * self.M)), CLIP_EXP_MIN, CLIP_EXP_MAX))
        alpha = Tr**(self.N * (self.M - 1)) * term_exp
        
        # Derivative of Alpha with respect to Tr (dAlpha/dTr)
        # dAlpha/dTr = Alpha * [ N(M-1)/Tr - L*N*M*Tr^(NM-1) ]
        term_bracket = (self.N * (self.M - 1) / Tr) - (self.L * self.N * self.M * Tr**(self.N * self.M - 1))
        dAlpha_dTr = alpha * term_bracket
        
        # da/dT = ac * (dAlpha/dTr) * (dTr/dT) = ac * dAlpha/dTr * (1/Tc)
        da_dT = self.a_c * dAlpha_dTr / self.Tc
        return da_dT

    def _solve_cubic_cardano(self, c2, c1, c0):
        p = c1 - c2**2 / 3.0
        q = c0 + (2.0 * c2**3 - 9.0 * c1 * c2) / 27.0
        delta = (q / 2.0)**2 + (p / 3.0)**3
        
        roots = []
        if delta >= 0:
            delta_sqrt = np.sqrt(np.clip(delta, 0.0, None))
            u = np.cbrt(-q / 2.0 + delta_sqrt)
            v = np.cbrt(-q / 2.0 - delta_sqrt)
            roots = [u + v]
        else:
            phi_arg = np.clip((-q / 2.0) / np.sqrt(np.clip(-(p / 3.0)**3, CLIP_LOG_MIN, None)), -1.0, 1.0)
            phi = np.arccos(phi_arg)
            sqrt_term = 2 * np.sqrt(np.clip(-p / 3.0, 0.0, None))
            roots = [
                sqrt_term * np.cos(phi / 3.0),
                sqrt_term * np.cos((phi + 2 * np.pi) / 3.0),
                sqrt_term * np.cos((phi + 4 * np.pi) / 3.0)
            ]
        return [t - c2 / 3.0 for t in roots]

    def tc_pr(self, T, v):
        """Calculates Pressure (Pa) given T and Vm."""
        a = self._get_a_param(T)
        # Correct generic cubic form: P = RT/(v+c-b) - a/((v+c+d)(v+c+e))
        # Note: b, d, e already include c shifts in their definitions? 
        # Reviewing definitions in __init__:
        # b = b_pr - c
        # d, e are roots of (v+c)^2 + 2b(v+c) - b^2.
        # The form used is Consistent PR.
        # P = RT/(v - b) - a/((v+d)(v+e)) where v is the shifted volume V_eos = V_real - c?
        # No, Le Guennec uses: P = RT / (v + c - b_pr) ...
        # Let's stick to the parameters defined in __init__ which are shifted.
        # Based on __init__: b, d, e are derived for the equation:
        # P = RT / (v_exp - b) - a / ((v_exp + d) * (v_exp + e)) 
        # where b, d, e are calculated to account for the translation c_si.
        
        term1 = R * T / (v - self.b)
        term2 = a / ((v + self.d) * (v + self.e))
        return term1 - term2

    def solve_eos_for_z(self, T, P):
        """Returns sorted physical Z roots (Z > B_dim)."""
        a = self._get_a_param(T)
        A = a * P / (R**2 * T**2)
        B = self.b * P / (R * T)
        D = self.d * P / (R * T)
        E = self.e * P / (R * T)
        
        c2 = E + D - B - 1
        c1 = A + E*D - (B + 1)*(E + D)
        c0 = -(A*B + E*D + B*E*D)

        if not np.all(np.isfinite([c2, c1, c0])): return tuple()
        roots = self._solve_cubic_cardano(c2, c1, c0)
        
        # B_dim is the dimensionless co-volume.
        # Physically Z must be such that V > b. 
        # Since Z = PV/RT and B = Pb/RT -> Z > B.
        return tuple(sorted([r for r in roots if r > B + 1e-9]))

    def calculate_psat(self, T):
        """Calculates saturation pressure."""
        if T >= self.Tc: return None
        
        Tr = T / self.Tc
        exp_arg = 5.37 * (1 + self.omega) * (1 - 1/Tr)
        p_guess = self.Pc * np.exp(np.clip(exp_arg, CLIP_EXP_MIN, CLIP_EXP_MAX))
        p_guess = max(1e2, min(p_guess, self.Pc * 0.99))
        
        def fugacity_diff(P):
            if P <= 0: return 1e9
            z_roots = self.solve_eos_for_z(T, P)
            if len(z_roots) < 2: return -1.0 if P > p_guess else 1.0
            
            Z_L, Z_V = min(z_roots), max(z_roots)
            phi_L = self.calculate_fugacity(Z_L, T, P)
            phi_V = self.calculate_fugacity(Z_V, T, P)
            return phi_L - phi_V

        try:
            return brentq(fugacity_diff, 1e2, self.Pc * 1.05, xtol=1e-5)
        except:
            return None

    def calculate_fugacity(self, Z, T, P):
        """Calculates ln(phi)."""
        a = self._get_a_param(T)
        A = a * P / (R**2 * T**2)
        B = self.b * P / (R * T)
        D = self.d * P / (R * T)
        E = self.e * P / (R * T)
        
        if abs(E - D) < 1e-12: return np.nan # Degenerate case
        
        # Eq: ln(phi) = Z - 1 - ln(Z-B) - A/(E-D) * ln((Z+E)/(Z+D))
        term1 = Z - 1.0
        term2 = np.log(np.clip(Z - B, 1e-50, None))
        term3 = (A / (E - D)) * np.log(np.clip((Z + E) / (Z + D), 1e-50, None))
        
        return term1 - term2 - term3

    # =========================================================================
    #  THERMODYNAMIC PROPERTIES (Enthalpy, Entropy)
    # =========================================================================

    def calculate_ideal_props(self, T, cp_params, T_ref=298.15):
        """
        Calculates Ideal Gas Enthalpy and Entropy relative to T_ref = 298.15 K.
        Cp_ig = A + B*T + C*T^2 + D*T^3 + E*T^4  (J/mol.K)
        """
        A, B, C, D, E = cp_params.get('A', 0), cp_params.get('B', 0), cp_params.get('C', 0), cp_params.get('D', 0), cp_params.get('E', 0)
        
        # Ideal Enthalpy: Integral(Cp dT) from T_ref to T
        def integ_H(t):
            return A*t + (B/2)*t**2 + (C/3)*t**3 + (D/4)*t**4 + (E/5)*t**5
        
        H_ig = integ_H(T) - integ_H(T_ref)
        
        # Ideal Entropy (Temperature part): Integral(Cp/T dT) from T_ref to T
        def integ_S(t):
            return A*np.log(t) + B*t + (C/2)*t**2 + (D/3)*t**3 + (E/4)*t**4
            
        S_ig_T = integ_S(T) - integ_S(T_ref)
        
        return H_ig, S_ig_T

    def calculate_residual_props(self, T, P, Z):
        """
        Calculates Residual Enthalpy and Entropy using the EOS.
        H_res = H_real - H_ideal
        S_res = S_real - S_ideal (at same T, P)
        """
        a = self._get_a_param(T)
        da_dT = self._da_dT(T)
        
        # Dimensionless parameters
        B = self.b * P / (R * T)
        D = self.d * P / (R * T)
        E = self.e * P / (R * T)
        
        if abs(E - D) < 1e-12: return 0.0, 0.0

        # Logarithmic term common to both
        log_term = np.log(np.clip((Z + E) / (Z + D), 1e-50, None))
        inv_denom = 1.0 / (self.e - self.d) # Dimensional denominator (E-D)*RT/P
        
        # Residual Enthalpy
        # H_res = RT(Z-1) + [(T*da/dT - a) / (e-d)] * ln(...)
        H_res = R * T * (Z - 1.0) + ((T * da_dT - a) * inv_denom) * log_term
        
        # Residual Entropy
        # S_res = R*ln(Z-B) + [da/dT / (e-d)] * ln(...)
        S_res = R * np.log(np.clip(Z - B, 1e-50, None)) + (da_dT * inv_denom) * log_term
        
        return H_res, S_res

    def calculate_properties(self, T, P, Z, cp_params):
        """
        Returns Total Enthalpy and Entropy (J/mol and J/mol.K).
        Reference state: Ideal Gas at T=298.15K, P=101325 Pa -> H=0, S=0 (hypothetical)
        """
        P_ref = 101325.0
        T_ref = 298.15
        
        # 1. Ideal Gas Contributions
        H_ig, S_ig_T = self.calculate_ideal_props(T, cp_params, T_ref)
        
        # Pressure correction for Ideal Entropy: -R*ln(P/P_ref)
        S_ig_P = -R * np.log(np.clip(P / P_ref, 1e-50, None))
        
        S_ideal_total = S_ig_T + S_ig_P
        
        # 2. Residual Contributions
        H_res, S_res = self.calculate_residual_props(T, P, Z)
        
        # 3. Totals
        H_total = H_ig + H_res
        S_total = S_ideal_total + S_res
        
        return H_total, S_total

    # =========================================================================
    #  INVERSE SOLVERS: FROM H, S -> T, P, V
    # =========================================================================

    def solve_from_HS(self, H_target, S_target, cp_params, phase='auto', T_guess=None, P_guess=None):
        """
        Solve for temperature (K) and pressure (Pa) given target molar enthalpy (J/mol)
        and entropy (J/mol.K). Returns dict with keys: success, T, P, Z, v, H, S.

        phase: 'auto', 'liq', or 'vap' to prefer liquid or vapor root when multiple Z exist.
        """
        # Initial guesses
        if T_guess is None:
            T_guess = min(max(300.0, self.Tc * 0.5), self.Tc * 0.9)
        if P_guess is None:
            P_guess = max(1e3, min(self.Pc * 0.5, 1e6))

        def residuals(x):
            T = float(x[0])
            lnP = float(x[1])
            P = np.exp(lnP)

            # avoid non-physical T
            if T <= 0 or not np.isfinite(T) or not np.isfinite(P):
                return [1e6, 1e6]

            z_roots = self.solve_eos_for_z(T, P)
            if len(z_roots) == 0:
                return [1e5, 1e5]

            # choose Z based on phase preference or by closeness
            if phase == 'liq' and len(z_roots) >= 1:
                Z = min(z_roots)
            elif phase == 'vap' and len(z_roots) >= 1:
                Z = max(z_roots)
            else:
                # choose root that gives properties closest to targets
                best = None
                best_err = 1e9
                for zr in z_roots:
                    H_calc, S_calc = self.calculate_properties(T, P, zr, cp_params)
                    err = abs(H_calc - H_target) + abs(S_calc - S_target)
                    if err < best_err:
                        best_err = err
                        best = (zr, H_calc, S_calc)
                Z, H_calc, S_calc = best

            H_calc, S_calc = self.calculate_properties(T, P, Z, cp_params)
            return [H_calc - H_target, S_calc - S_target]

        x0 = [T_guess, np.log(P_guess)]
        try:
            sol = root(residuals, x0, method='hybr', tol=1e-6)
        except Exception:
            return {'success': False, 'message': 'Solver crashed'}

        if not sol.success:
            return {'success': False, 'message': sol.message}

        T_sol = float(sol.x[0])
        P_sol = float(np.exp(sol.x[1]))
        z_roots = self.solve_eos_for_z(T_sol, P_sol)
        if len(z_roots) == 0:
            return {'success': False, 'message': 'No EOS roots at solution'}

        if phase == 'liq':
            Z_sol = min(z_roots)
        elif phase == 'vap':
            Z_sol = max(z_roots)
        else:
            # pick closest to S_target
            best = min(z_roots, key=lambda zr: abs(self.calculate_properties(T_sol, P_sol, zr, cp_params)[1] - S_target))
            Z_sol = best

        H_sol, S_sol = self.calculate_properties(T_sol, P_sol, Z_sol, cp_params)
        v_molar = Z_sol * R * T_sol / P_sol

        return {
            'success': True,
            'T': T_sol,
            'P': P_sol,
            'Z': Z_sol,
            'v': v_molar,
            'H': H_sol,
            'S': S_sol,
            'message': sol.message
        }

    def solve_from_PH(self, P_target, H_target, cp_params, phase='auto', T_guess=None):
        """
        Solve for T (K) given pressure P (Pa) and molar enthalpy H (J/mol).
        Returns dict similar to solve_from_HS.
        """
        if T_guess is None:
            T_guess = max(0.5 * self.Tc, 300.0)

        def res(x):
            T = float(x[0])
            if T <= 0 or not np.isfinite(T):
                return [1e6]
            z_roots = self.solve_eos_for_z(T, P_target)
            if not z_roots:
                return [1e5]

            if phase == 'liq':
                Z = min(z_roots)
            elif phase == 'vap':
                Z = max(z_roots)
            else:
                # choose root with closest H
                best = None
                best_err = 1e12
                for zr in z_roots:
                    Hc, Sc = self.calculate_properties(T, P_target, zr, cp_params)
                    err = abs(Hc - H_target)
                    if err < best_err:
                        best_err = err
                        best = (zr, Hc)
                Z, Hc = best

            H_calc, S_calc = self.calculate_properties(T, P_target, Z, cp_params)
            return [H_calc - H_target]

        from scipy.optimize import root
        try:
            sol = root(res, [T_guess], method='hybr', tol=1e-6)
        except Exception:
            return {'success': False, 'message': 'Solver crashed'}

        if not sol.success:
            return {'success': False, 'message': sol.message}

        T_sol = float(sol.x[0])
        z_roots = self.solve_eos_for_z(T_sol, P_target)
        if len(z_roots) == 0:
            return {'success': False, 'message': 'No EOS roots at solution'}

        if phase == 'liq':
            Z_sol = min(z_roots)
        elif phase == 'vap':
            Z_sol = max(z_roots)
        else:
            best = min(z_roots, key=lambda zr: abs(self.calculate_properties(T_sol, P_target, zr, cp_params)[0] - H_target))
            Z_sol = best

        H_sol, S_sol = self.calculate_properties(T_sol, P_target, Z_sol, cp_params)
        v_molar = Z_sol * R * T_sol / P_target
        return {'success': True, 'T': T_sol, 'P': P_target, 'Z': Z_sol, 'v': v_molar, 'H': H_sol, 'S': S_sol, 'message': sol.message}

    def solve_from_TS(self, T_target, S_target, cp_params, phase='auto', P_guess=None):
        """
        Solve for Pressure (Pa) given temperature T (K) and molar entropy S (J/mol.K).
        Returns dict with keys similar to other solvers.
        """
        if P_guess is None:
            P_guess = max(1e3, self.Pc * 0.1)

        def res(x):
            lnP = float(x[0])
            P = np.exp(lnP)
            if P <= 0 or not np.isfinite(P):
                return [1e6]
            z_roots = self.solve_eos_for_z(T_target, P)
            if not z_roots:
                return [1e5]

            if phase == 'liq':
                Z = min(z_roots)
            elif phase == 'vap':
                Z = max(z_roots)
            else:
                best = None
                best_err = 1e12
                for zr in z_roots:
                    Hc, Sc = self.calculate_properties(T_target, P, zr, cp_params)
                    err = abs(Sc - S_target)
                    if err < best_err:
                        best_err = err
                        best = (zr, Sc)
                Z, S_calc_guess = best

            H_calc, S_calc = self.calculate_properties(T_target, P, Z, cp_params)
            return [S_calc - S_target]

        from scipy.optimize import root
        try:
            sol = root(res, [np.log(P_guess)], method='hybr', tol=1e-6)
        except Exception:
            return {'success': False, 'message': 'Solver crashed'}

        if not sol.success:
            return {'success': False, 'message': sol.message}

        P_sol = float(np.exp(sol.x[0]))
        z_roots = self.solve_eos_for_z(T_target, P_sol)
        if len(z_roots) == 0:
            return {'success': False, 'message': 'No EOS roots at solution'}

        if phase == 'liq':
            Z_sol = min(z_roots)
        elif phase == 'vap':
            Z_sol = max(z_roots)
        else:
            best = min(z_roots, key=lambda zr: abs(self.calculate_properties(T_target, P_sol, zr, cp_params)[1] - S_target))
            Z_sol = best

        H_sol, S_sol = self.calculate_properties(T_target, P_sol, Z_sol, cp_params)
        v_molar = Z_sol * R * T_target / P_sol
        return {'success': True, 'T': T_target, 'P': P_sol, 'Z': Z_sol, 'v': v_molar, 'H': H_sol, 'S': S_sol, 'message': sol.message}

    def saturation_entropies(self, T):
        """
        For a given temperature below Tc, returns saturation pressure and
        liquid/vapor molar entropies and enthalpies when biphasic.
        Returns dict with keys: Psat, s_liq, s_vap, h_liq, h_vap or None if not biphasic.
        """
        Psat = self.calculate_psat(T)
        if Psat is None:
            return None

        z_roots = self.solve_eos_for_z(T, Psat)
        if len(z_roots) < 2:
            return None

        Z_L, Z_V = min(z_roots), max(z_roots)
        h_l, s_l = self.calculate_properties(T, Psat, Z_L, {})
        h_v, s_v = self.calculate_properties(T, Psat, Z_V, {})
        return {'Psat': Psat, 's_liq': s_l, 's_vap': s_v, 'h_liq': h_l, 'h_vap': h_v}

    # =========================================================================
    #  JSON helpers for ideal Cp parameters
    # =========================================================================

    @staticmethod
    def get_cp_params_from_json(substance_name, json_path='tc_pr_substances.json'):
        """
        Check `json_path` for ideal Cp parameters for `substance_name`.
        Expected format inside the substance entry is a dict under key 'ideal_cp'
        containing keys 'A','B','C','D','E' (optional). Returns dict or None.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return None

        entry = data.get(substance_name)
        if entry is None:
            # try searching by 'name' field
            for k, v in data.items():
                if isinstance(v, dict) and v.get('name') == substance_name:
                    entry = v
                    break

        if not entry:
            return None

        cp = entry.get('ideal_cp') or entry.get('cp_ideal') or entry.get('Cp') or entry.get('cp')
        if isinstance(cp, dict):
            # ensure numeric values
            return {k: float(v) for k, v in cp.items() if k in ('A','B','C','D','E')}
        return None

    @staticmethod
    def save_cp_params_to_json(substance_name, cp_params, json_path='tc_pr_substances.json'):
        """
        Save cp_params (dict with keys A..E) for substance_name into JSON file.
        If substance not present, it will create a new entry with only 'name' and 'ideal_cp'.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}

        if substance_name in data:
            entry = data[substance_name]
        else:
            # try to find by 'name' field
            entry = None
            for k, v in data.items():
                if isinstance(v, dict) and v.get('name') == substance_name:
                    entry = v
                    break
            if entry is None:
                data[substance_name] = {'name': substance_name}
                entry = data[substance_name]

        entry['ideal_cp'] = {k: float(cp_params.get(k, 0.0)) for k in ('A','B','C','D','E')}

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
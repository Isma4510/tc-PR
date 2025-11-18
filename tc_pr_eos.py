# tc_pr_eos.py

import numpy as np
from scipy.optimize import brentq
import warnings

# Universal gas constant (J/(mol*K))
R = 8.31446261815324

# Safety limits for numerical stability
CLIP_EXP_MIN = -700.0
CLIP_EXP_MAX = 700.0
CLIP_LOG_MIN = 1e-300

class TC_PR_EOS:
    """
    Twu-Coon-Peng-Robinson (tc-PR) Equation of State.
    
    References:
    - Le Guennec et al. (2016): Accurate modeling of high-pressure phase equilibria...
    - Twu et al. (1991): A generalized vapor pressure equation...
    """
    def __init__(self, substance_params):
        """
        Initialize the EOS with substance parameters.
        
        Args:
            substance_params (dict): Dictionary containing:
                - 'Tc': Critical Temperature (K)
                - 'Pc': Critical Pressure (Pa)
                - 'omega': Acentric factor
                - 'L', 'M', 'N': Twu alpha function parameters
                - 'c' (optional): Peneloux volume translation parameter (m3/mol)
                  If 'c' is missing or None, it is calculated from omega.
        """
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
        # b is corrected by volume translation c
        self.b = 0.07780 * (R * self.Tc) / self.Pc - self.c_si
        
        if self.b <= 0:
            raise ValueError(f"Invalid 'b' parameter (<=0) for {self.params.get('name', 'substance')}.")

        # Parameters for the cubic equation solver (shifted for volume translation)
        sqrt_term_inner = (2 * self.b + 4 * self.c_si)**2 - 8 * self.c_si**2 + 4 * self.b**2
        if sqrt_term_inner < 0:
             raise ValueError("Invalid parameters: negative value in sqrt for d/e calculation.")
        
        sqrt_term = np.sqrt(sqrt_term_inner) 
        self.e = ((2 * self.b + 4 * self.c_si) + sqrt_term) / 2
        self.d = ((2 * self.b + 4 * self.c_si) - sqrt_term) / 2

    def _calculate_c_from_omega(self):
        """Calculates volume translation parameter c if not provided."""
        return (R * self.Tc / self.Pc) * (-0.0065 + 0.0198 * self.omega)

    def _calculate_twu91_alpha(self, T):
        """Calculates the alpha(T) function using Twu (1991) parameters."""
        if self.L is None or self.M is None or self.N is None:
            raise ValueError(f"L, M, N parameters missing for {self.params.get('name', 'substance')}")
        
        Tr = T / self.Tc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exp_arg = self.L * (1 - Tr**(self.N * self.M))
            exp_term = np.exp(np.clip(exp_arg, CLIP_EXP_MIN, CLIP_EXP_MAX))
            alpha = Tr**(self.N * (self.M - 1)) * exp_term
        return alpha

    def _get_a_param(self, T):
        """Returns the attractive parameter a(T)."""
        return self.a_c * self._calculate_twu91_alpha(T)

    def _solve_cubic_cardano(self, c2, c1, c0):
        """
        Solves the cubic equation Z^3 + c2*Z^2 + c1*Z + c0 = 0 using Cardano's method.
        Returns real roots.
        """
        p = c1 - c2**2 / 3.0
        q = c0 + (2.0 * c2**3 - 9.0 * c1 * c2) / 27.0
        delta = (q / 2.0)**2 + (p / 3.0)**3
        
        roots = []
        if delta >= 0:
            delta_sqrt = np.sqrt(np.clip(delta, 0.0, None))
            u_arg = -q / 2.0 + delta_sqrt
            v_arg = -q / 2.0 - delta_sqrt
            u = np.cbrt(u_arg)
            v = np.cbrt(v_arg)
            t1 = u + v
            roots = [t1]
        else:
            phi_arg_inner = -(p / 3.0)**3
            phi_arg_num = -q / 2.0
            phi_arg_denom = np.sqrt(np.clip(phi_arg_inner, CLIP_LOG_MIN, None))
            phi_arg = np.clip(phi_arg_num / phi_arg_denom, -1.0, 1.0)
            
            phi = np.arccos(phi_arg)
            sqrt_term_inner = -p / 3.0
            sqrt_term = 2 * np.sqrt(np.clip(sqrt_term_inner, CLIP_LOG_MIN, None))
            
            t1 = sqrt_term * np.cos(phi / 3.0)
            t2 = sqrt_term * np.cos((phi + 2 * np.pi) / 3.0)
            t3 = sqrt_term * np.cos((phi + 4 * np.pi) / 3.0)
            roots = [t1, t2, t3]
            
        return [t - c2 / 3.0 for t in roots]
    
    def tc_pr(self, T, v):
        """
        Calculates pressure P using the tc-PR EOS for given T and molar volume v.
        """
        a = self._get_a_param(T)
        term1 = R * T / (v + self.c_si - self.b)
        term2 = a / ((v + self.c_si + self.d) * (v + self.c_si + self.e))
        P = term1 - term2
        return P

    def solve_eos_for_z(self, T, P, a=None, b=None, d=None, e=None):
        """
        Solves the EOS for the compressibility factor Z at given T and P.
        Can optionally override parameters a, b, d, e (useful for mixtures).
        Returns a tuple of sorted physical roots (Z > B).
        """
        a = a if a is not None else self._get_a_param(T)
        b = b if b is not None else self.b
        d = d if d is not None else self.d
        e = e if e is not None else self.e
        
        A = a * P / (R**2 * T**2)
        B = b * P / (R * T)
        D = d * P / (R * T)
        E = e * P / (R * T)
        
        c2 = E + D - B - 1
        c1 = A + E*D - (B + 1)*(E + D)
        c0 = -(A*B + E*D + B*E*D)

        if not all(np.isfinite([c2, c1, c0])): return tuple()
        roots = self._solve_cubic_cardano(c2, c1, c0)
        
        # Filter unphysical roots (Z must be greater than B)
        return tuple(sorted([r for r in roots if r > B]))

    def calculate_fugacity_coeff(self, Z, T, P):
        """
        Calculates ln(phi) for a pure component given a specific Z root.
        """
        a = self._get_a_param(T) 
        A = a * P / (R**2 * T**2) 
        B = self.b * P / (R * T) 
        D = self.d * P / (R * T) 
        E = self.e * P / (R * T) 

        if abs(E - D) < 1e-12 or (Z - B) <= 0 or (Z + E) <= 0 or (Z + D) <= 0: 
            return np.nan
        
        log_term_1 = np.log(Z - B) 
        log_term_2_arg = (Z + E) / (Z + D)
        if log_term_2_arg <= 0: return np.nan
        log_term_2 = np.log(log_term_2_arg)
        
        ln_phi = Z - 1 - log_term_1 - (A / (E - D)) * log_term_2 
        return ln_phi

    def calculate_psat(self, T):
        """
        Calculates the saturation pressure at Temperature T.
        Returns P_sat (Pa) or None if T > Tc or no solution found.
        """
        if T >= self.Tc: return None
        
        # Initial guess using Wilson correlation
        Tr = T / self.Tc
        exp_arg = 5.37 * (1 + self.omega) * (1 - 1/Tr)
        p_guess = self.Pc * np.exp(np.clip(exp_arg, CLIP_EXP_MIN, CLIP_EXP_MAX))
        
        # Constraint guess within physical bounds
        p_guess = max(1e2, min(p_guess, self.Pc * 0.99))
        
        def fugacity_difference(P):
            if P <= 0: return 1.0
            z_roots = self.solve_eos_for_z(T, P)
            # Need at least liquid and vapor roots
            if len(z_roots) < 2: 
                return -1.0 if P > p_guess else 1.0
            
            Z_liq, Z_vap = min(z_roots), max(z_roots)
            ln_phi_liq = self.calculate_fugacity_coeff(Z_liq, T, P)
            ln_phi_vap = self.calculate_fugacity_coeff(Z_vap, T, P)
            
            if np.isnan(ln_phi_liq) or np.isnan(ln_phi_vap): 
                return -1.0 if P > p_guess else 1.0
            
            return ln_phi_liq - ln_phi_vap

        try:
            return brentq(fugacity_difference, 1e2, self.Pc * 1.05, xtol=1e-8, rtol=1e-6, maxiter=150)
        except (ValueError, RuntimeError):
            return None

    def get_psat_proxy(self, T):
        """
        Attempts to calculate Psat. If T > Tc (supercritical), returns Pc.
        Useful for continuity in mixture algorithms.
        """
        P_sat = self.calculate_psat(T)
        if P_sat is None:
            return self.Pc 
        return P_sat
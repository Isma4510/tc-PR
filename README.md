# tc-PR Thermodynamics Calculator

A Python implementation of the **Translated-Consistent-Peng-Robinson (tc-PR)** Equation of State (EoS). This project provides accurate thermodynamic property calculations (Saturation Pressure, Density, Compressibility Factor) for a wide range of pure fluids, including common gases, hydrocarbons, and refrigerants.

It features a Graphical User Interface (GUI) for interaction.

## Features

* Uses the Peng-Robinson equation with the Twu (1991) alpha function and Peneloux volume translation for precise saturation pressure and liquid density predictions.
* **Graphical User Interface (GUI):**
    * **Saturation Plotter:** Visualize $P_{sat}$ vs $T$ curves instantly.
    * **Flexible Solver:** Calculate properties from any pair of inputs: $(T, P) \rightarrow V, Z$, $(P, V) \rightarrow T, Z$, etc.
    * **Fluid Search:** Quickly find fluids in the database.
* **Extensive Database:** Includes parameters ($T_c, P_c, \omega, L, M, N$) for over **100 fluids** (JSON format).
* **Robust Algorithms:** Implements Cardano's method for cubic roots and Brent's method for saturation pressure iteration.

## Installation

1.  **Clone or Download** this repository.
2.  Ensure you have **Python 3.x** installed.
3.  Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
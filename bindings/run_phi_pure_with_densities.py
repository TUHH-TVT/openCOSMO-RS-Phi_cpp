import copy as cp
from pathlib import Path

import numpy as np
import openCOSMORS


SCRIPT_DIR = Path(__file__).resolve().parent

options = {
    "sw_skip_COSMOSPACE_errors": 0,
    "RelaxFactor": 1,
    "sw_SR_model": "COSMORS_TUHH",
    "sw_SR_COSMOfiles_type": "ORCA_COSMO_TZVPD",
    "sw_SR_combTerm": 3,
    "sw_SR_alwaysReloadSigmaProfiles": 0,
    "sw_SR_alwaysCalculateSizeRelatedParameters": 0,
    "sw_SR_useSegmentReferenceStateForInteractionMatrix": 0,
    "sw_SR_calculateContactStatisticsAndAdditionalProperties": 0,
    "sw_SR_partialInteractionMatrices": [],
    "sw_SR_atomicNumber": 0,
    "sw_SR_misfit": 0,
    "sw_SR_differentiateHydrogens": 0,
    "sw_SR_differentiateMoleculeGroups": 0,
    "sw_residual_properties": 1,
    "sw_phi": 1,
    "sw_PT_Flash": 0,
    "sw_PX_Flash": 0,
    "sw_Pbub": 0,
    "sw_Pdew": 0,
    "sw_Tbub": 0,
    "sw_Tdew": 0,
    "sw_lnGamma": 0,
    "sw_checkAndCalcBinaryLLE": 0,
    "sw_isotherm": 0,
    "sw_excess": 0,
    "sw_criticalPoint": 0,
    "sw_criticalPoint_specifiedx": 0,
    "sw_azeotropicPoint": 0,
    "sw_Psat": 0,
}

parameters = {
    "Aeff": 4.857011,
    "ln_alpha": 0.3121634,
    "ln_CHB": 0.1482344,
    "CHBT": 0.0,
    "SigmaHB": 0.009443451,
    "Rav": 0.5,
    "RavCorr": 1.1,
    "fCorr": 2.4,
    "comb_SG_z_coord": 0.0,
    "comb_SG_A_std": 35.31765,
    "comb_modSG_exp": 0.6666666666666666,
    "comb_lambda0": 0.463,
    "comb_lambda1": 0.42,
    "comb_lambda2": 0.065,
    "comb_SGG_lambda": 0.773,
    "comb_SGG_beta": 0.778,
    "hole_area": 12.57,
    "pure_component_parameters": [],
    "radii": {},
    "exp": {
        "EigenMatrixToFile": 0,
        "dG_eta": -4.211934,
        "dG_omega_ring": 0.2882878,
        "dG_vdw_1": 0.03761312,
        "dG_vdw_6": 0.01778966,
        "dG_vdw_7": 0.0006285596,
        "dG_vdw_8": 0.005266507,
        "dG_vdw_9": 0.00537074,
        "dG_vdw_14": 0.00331679,
        "dG_vdw_15": 0.00397561,
        "dG_vdw_16": 0.03259574,
        "dG_vdw_17": 0.03610742,
        "dG_vdw_35": 0.044093,
        "dG_vdw_53": 0.2044762,
    },
}

ATOMIC_MASSES_G_PER_MOL = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "P": 30.973762,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
}


def add_calculation_to_structure(components, temperatures, pressures, calc_type, component_indices):
    number_of_concentrations = len(temperatures)
    pressure_array = np.zeros((number_of_concentrations * 2, 1), np.float64)
    temperature_array = np.zeros(number_of_concentrations * 2, np.float64)

    for j in range(number_of_concentrations):
        pressure_array[j, 0] = pressures[j]
        pressure_array[j + number_of_concentrations, 0] = pressures[j]
        temperature_array[j] = temperatures[j]
        temperature_array[j + number_of_concentrations] = temperatures[j]

    return {
        "overall_concentrations": np.ones((number_of_concentrations, 1), np.float64),
        "concentrations": np.ones((number_of_concentrations * 2, 1), np.float64),
        "temperatures": temperature_array,
        "molar_volumes": np.zeros((number_of_concentrations * 2, 1), np.float64),
        "total_pressure": pressure_array,
        "target_pressure": cp.deepcopy(pressure_array),
        "attractive_pressure": np.zeros((number_of_concentrations * 2, 1), np.float64),
        "repulsive_pressure": np.zeros((number_of_concentrations * 2, 1), np.float64),
        "components": components,
        "reference_state_types": np.ones((number_of_concentrations * 2,), np.int64) * 5,
        "reference_state_concentrations": np.zeros((number_of_concentrations * 2, 0), np.float64),
        "type": calc_type,
        "component_indices": component_indices,
    }


def fill_missing_calculation_structures(calculations):
    for i, calculation in enumerate(calculations):
        number_of_components = calculation["concentrations"].shape[1]
        number_of_concentrations = calculation["concentrations"].shape[0]
        number_of_components_with_hole = number_of_components + 1

        calculation["chemical_Potentials"] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation["ln_phi_attractive"] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation["ln_phi_repulsive"] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation["ln_phi"] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation["ln_gamma_x_SR_combinatorial_calc"] = np.zeros(
            (number_of_concentrations, number_of_components_with_hole), np.float64
        )
        calculation["ln_gamma_x_SR_residual_calc"] = np.zeros(
            (number_of_concentrations, number_of_components_with_hole), np.float64
        )
        calculation["ln_gamma_x_SR_calc"] = np.zeros(
            (number_of_concentrations, number_of_components_with_hole), np.float64
        )
        calculation["hres"] = np.zeros((number_of_concentrations, 1), np.float64)
        calculation["cpres"] = np.zeros((number_of_concentrations, 1), np.float64)
        calculation["index"] = i

    return calculations


def molar_mass_kg_per_mol_from_orca_cosmo(path):
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "#XYZ_FILE":
            atom_count = int(lines[i + 1].strip())
            atom_lines = lines[i + 3 : i + 3 + atom_count]
            molar_mass_g_per_mol = 0.0
            for atom_line in atom_lines:
                symbol = atom_line.split()[0]
                molar_mass_g_per_mol += ATOMIC_MASSES_G_PER_MOL[symbol]
            return molar_mass_g_per_mol / 1000.0

    raise ValueError(f"Could not find #XYZ_FILE atom block in {path}")


def calculate_vapor_pressure_and_densities(orcacosmo_file, pure_component_parameters, temperatures, pressure_guess=101325.0):
    """Calculate pure-component vapor pressure plus vapor/liquid densities for each temperature.

    Args:
        orcacosmo_file: Path to the pure component ORCA COSMO file.
        pure_component_parameters: Component EoS parameters as [b_i, delta_m0, delta_mT, b_h]
            or as [[b_i, delta_m0, delta_mT, b_h]].
        temperatures: Iterable of temperatures in K.
        pressure_guess: Initial pressure guess in Pa.

    Returns:
        A list of dictionaries, one per temperature.
    """
    component_path = Path(orcacosmo_file)
    if not component_path.is_absolute():
        component_path = SCRIPT_DIR / component_path

    components = [str(component_path)]
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.ndim != 1:
        raise ValueError("temperatures must be a one-dimensional array or list")

    pure_component_parameters = np.asarray(pure_component_parameters, dtype=np.float64)
    if pure_component_parameters.ndim == 1:
        pure_component_parameters = pure_component_parameters.reshape(1, -1)

    parameters["pure_component_parameters"] = pure_component_parameters
    calculation = add_calculation_to_structure(
        components=components,
        temperatures=temperatures,
        pressures=np.full(len(temperatures), pressure_guess, np.float64),
        calc_type="LL",
        component_indices=[0],
    )
    calculations = fill_missing_calculation_structures([calculation])

    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)

    result = calculations[0]
    molar_mass = molar_mass_kg_per_mol_from_orca_cosmo(components[0])
    number_of_temperatures = len(temperatures)

    rows = []
    for i, temperature in enumerate(temperatures):
        vapor_index = i
        liquid_index = i + number_of_temperatures

        vapor_pressure = float(result["total_pressure"][liquid_index][0])
        vapor_molar_volume = float(result["molar_volumes"][vapor_index][0])
        liquid_molar_volume = float(result["molar_volumes"][liquid_index][0])
        vapor_molar_density = 1.0 / vapor_molar_volume
        liquid_molar_density = 1.0 / liquid_molar_volume

        rows.append(
            {
                "temperature_K": float(temperature),
                "vapor_pressure_Pa": vapor_pressure,
                "vapor_molar_volume_m3_per_mol": vapor_molar_volume,
                "liquid_molar_volume_m3_per_mol": liquid_molar_volume,
                "vapor_molar_density_mol_per_m3": vapor_molar_density,
                "liquid_molar_density_mol_per_m3": liquid_molar_density,
                "vapor_mass_density_kg_per_m3": vapor_molar_density * molar_mass,
                "liquid_mass_density_kg_per_m3": liquid_molar_density * molar_mass,
            }
        )

    return rows


if __name__ == "__main__":
    try:
        methanol_pc_parameters = [56.08186814344738, 1.3518314651065213, 3.3225493224786162, 8.713083189565843]
        methanol_temperatures = np.array([280.0, 298.15, 320.0], np.float64)
        results = calculate_vapor_pressure_and_densities(
            "methanol.orcacosmo",
            methanol_pc_parameters,
            methanol_temperatures,
        )

        print("Methanol vapor pressure and saturated phase densities:")
        print(
            "T / K, Psat / Pa, vapor density / kg/m^3, liquid density / kg/m^3, "
            "vapor molar density / mol/m^3, liquid molar density / mol/m^3"
        )
        for row in results:
            print(
                f"{row['temperature_K']}, "
                f"{row['vapor_pressure_Pa']}, "
                f"{row['vapor_mass_density_kg_per_m3']}, "
                f"{row['liquid_mass_density_kg_per_m3']}, "
                f"{row['vapor_molar_density_mol_per_m3']}, "
                f"{row['liquid_molar_density_mol_per_m3']}"
            )

    except Exception as err:
        print("ERROR:")
        print(type(err))
        print(err.args)

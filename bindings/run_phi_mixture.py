
from calendar import TUESDAY
from subprocess import list2cmdline
import openCOSMORS
import numpy as np
import copy as cp
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import copy

options = {
    "sw_skip_COSMOSPACE_errors": 0,                                     # whether to skip COSMOSPACE errors in the case a parameter makes the equations unsolvable
    "RelaxFactor": 1,

    # short-range switches
    "sw_SR_model": "COSMORS_TUHH",                                      # later 'ideal', '15', '17FINE', etc.
    "sw_SR_COSMOfiles_type": "ORCA_COSMO_TZVPD",                        # or 'ORCA_COSMO_TZVPD'
    "sw_SR_combTerm": 3,

    # optional calculation switches
    "sw_SR_alwaysReloadSigmaProfiles": 0,
    "sw_SR_alwaysCalculateSizeRelatedParameters": 0,
    "sw_SR_useSegmentReferenceStateForInteractionMatrix": 0,            # 0 : conductor | 1 : pure segment
    "sw_SR_calculateContactStatisticsAndAdditionalProperties": 0,       # 0 : do not calculate
                                                                        # 1 : calculate contact statistics and average surface energies
                                                                        # 2 : calculate contact statistics, average surface energies and partial molar properties
    "sw_SR_partialInteractionMatrices": [],                             #['E_mf', 'G_hb']

    # segment descriptor switches
    "sw_SR_atomicNumber": 0,                                            # 0/1 : use atomic number or not
    "sw_SR_misfit": 0,                                                  # 0 : do not use misfit correlation
                                                                        # 1 : use misfit correlation on all molecules
                                                                        # 2 : use misfit correlation only on neutral molecules
    "sw_SR_differentiateHydrogens": 0,
    "sw_SR_differentiateMoleculeGroups": 0,

    # calculate Gamma Derivatives or Not
    "sw_residual_properties": 0,

    # openCOSMO-RS-Phi-related switches
    "sw_phi": 1,                                                        # 0/1: use openCOSMO-RS-Phi or not
    
    # Note: Of the following switches only one should be set to 1 at any time
    "sw_PT_Flash": 0,                                                   # 1: pT-Flash for mixture
    "sw_PX_Flash": 0,                                                   # 1: pX-Flash for mixture (not yet programmed)
    "sw_Pbub": 0,                                                       # 1: Bubble Pressure calculation for mixture
    "sw_Pdew": 0,                                                       # 1: Dew Pressure calculation for mixture
    "sw_Tbub": 0,                                                       # 1: Boiling Temperature calculation for mixture
    "sw_Tdew": 0,                                                       # 1: Dew Temperature calculation for mixture
    "sw_lnGamma": 0,                                                    # 1: Calculate activity coefficients (using liquid-like root)
    "sw_checkAndCalcBinaryLLE": 0,
    "sw_isotherm": 0,                                                   # 1: Calculate a specific isotherm for mixture and provide results in .txt file
    "sw_excess": 0,                                                     # 1: Calculate excess and mixing properties
    "sw_criticalPoint": 0,                                              # 1: Calculate critical point of a mixture for specified T
    'sw_criticalPoint_specifiedx': 0,                                   # 1: Calculate critical point of a mixture for specified x
    'sw_azeotropicPoint': 0,                                            # 1: Calculate azeotropic point of a mixture for specified T
    "sw_Psat": 0,
    
}

#aeff = 3.14159265359 * 1.15666741903236**2

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
    
    # Pure Components Parameters for openCOSMO-RS-Phi [[bi_comp1, delta_m0_comp1, delta_mT_comp1, bh_comp1], [bi_comp2, delta_m0_comp2, delta_mT_comp2, bh_comp2], ...]
    "pure_component_parameters":[[50.826951478727125, 0.5943262353122482, 3.1858544427725564, 10.669150253967132], [73.48869746828207, 0.8031718195904861, 3.548049119383619, 12.683442985210894]],
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
    }
}


def addCalculationToStructure(components, numberOfConcentrations, specified_temperatures, specified_pressures, calc_type, component_indices):
    number_of_concentrations = numberOfConcentrations
    pressures = np.zeros((number_of_concentrations * 2,1))
    temperatures = np.zeros(number_of_concentrations * 2)
    for j in range(number_of_concentrations):
        pressures[j, 0] = specified_pressures[j]
        pressures[j + number_of_concentrations, 0] = specified_pressures[j]
        temperatures[j] = specified_temperatures[j]
        temperatures[j + number_of_concentrations] = specified_temperatures[j]
    calculation = {
        'overall_concentrations' : np.array([[0.65, 0.35]]),
        'concentrations' : np.array([[0.65, 0.35], [0.65, 0.35]]),
        'temperatures' : temperatures,
        'molar_volumes' : np.zeros((number_of_concentrations * 2, 1)),
        'total_pressure': pressures,
        'target_pressure': cp.deepcopy(pressures),
        'attractive_pressure': np.zeros((number_of_concentrations * 2, 1)),
        'repulsive_pressure': np.zeros((number_of_concentrations * 2, 1)),
        'components' : components,
        'reference_state_types' : np.ones((number_of_concentrations*2,),np.int64)*5,
        'reference_state_concentrations' : np.zeros((number_of_concentrations*2, 0), np.float64),

        'type':           calc_type,

        'component_indices' : component_indices
    }
    return calculation


# fill missing fields for each calculation
def fill_missing_calculation_structures(calculations, options):

    for i, calculation in enumerate(calculations):

        number_of_components = calculation['concentrations'].shape[1]
        number_of_concentrations = calculation['concentrations'].shape[0]

        calculation['chemical_Potentials'] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation['ln_phi_attractive'] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation['ln_phi_repulsive'] = np.zeros((number_of_concentrations, number_of_components), np.float64)
        calculation['ln_phi'] = np.zeros((number_of_concentrations, number_of_components), np.float64)

        if(options['sw_phi'] == 1): 
            number_of_components += 1

        calculation['ln_gamma_x_SR_combinatorial_calc'] = np.zeros((number_of_concentrations, number_of_components),dtype= np.float64)
        calculation['ln_gamma_x_SR_residual_calc'] = np.zeros((number_of_concentrations, number_of_components), dtype=np.float64)
        calculation['ln_gamma_x_SR_calc'] = np.zeros((number_of_concentrations, number_of_components), dtype=np.float64)

        if(options['sw_excess'] == 1):
            calculation['gE'] = np.zeros((number_of_concentrations, 1), dtype=np.float64) 
            calculation['hE'] = np.zeros((number_of_concentrations, 1), dtype=np.float64) 
            calculation['cpE'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)
            calculation['gM'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)
            calculation['hM'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)
            calculation['cpM'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)

        if(options['sw_criticalPoint'] == 1 or options['sw_azeotropicPoint'] == 1 or options['sw_criticalPoint_specifiedx'] == 1 ):
            calculation['critical_T'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)
            calculation['critical_P'] = np.zeros((number_of_concentrations, 1), dtype=np.float64)
            calculation['critical_x'] = np.zeros((number_of_concentrations, number_of_components - 1), dtype=np.float64)

        calculation['hres'] = np.zeros((number_of_concentrations,1), dtype=np.float64)
        calculation['cpres'] = np.zeros((number_of_concentrations,1), dtype=np.float64)

        calculation['index'] = i

        
    return calculations


def extract_formula(path):
    parts = path.split('\\')
    for i, part in enumerate(parts):
        if part == 'NeutralSpecies':
            next_part = parts[i+1]
            formula_parts = next_part.split('_')[:2] 
            formula = '_'.join(formula_parts)
            return formula


def calculateCriticalOrAzeotropicPoint(components, pc_parameters, temperature, mode):
    if mode == "critical":
        options['sw_criticalPoint'] = 1
    else:
        options['sw_azeotropicPoint'] = 1

    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)

    number_of_concentrations = 1
    number_of_components = len(components)
    temperatures = [temperature]
    pressures = [101325]
    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, 'VL', [0, 1])
    array1 = np.zeros((number_of_concentrations, number_of_components))
    array2 = np.zeros((number_of_concentrations * 2, number_of_components))
    for i in range(number_of_concentrations):
        array1[i][0] = 0.0
        array1[i][1] = 1.0
        array2[i][0] = 0.0
        array2[i][1] = 1.0
        array2[i + number_of_concentrations][0] = 0.0
        array2[i + number_of_concentrations][1] = 1.0
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2
    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)

    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)

    print(calculations)


def VLE_isoT(components, pc_parameters, temperature, x1_range, calcnum, exp_data = [], DIPPR_Psat = [], PC_Psat = []):

    '''
    Calculates the isothermic VLE for a given BINARY mixture at a given temperature.

            Parameters:
                    components (list): list of paths to COSMO files of the mixture components
                    pc_parameters (list of lists): list of lists containing the pure component parameters of the mixture components
                    temperature (float): specified temperature in K
                    x_range (list): range of x1 in which the VLE is calculated ([x1_low, x1_high])
                    calcnum (int): number of calculations point, example: x_range = [0.0, 1.0], calcnum = 100 -> Delta x = 0.01
                    exp_data (list of lists): possibility to provide experimental data if available, scheme: [[x_values], [y_values], [pressures]]
                    DIPPR_Psat (list of floats): experimental argument: you can provide and later DIPPR_Psat for comparision with openCOSMO-RS-Phi
                    PC_Psat (list of floats:) s.o.

            Output:
                    plot and json file for given system
    '''
    
    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)

    number_of_components = len(components)
    comp1 = extract_formula(components[0])
    comp2 = extract_formula(components[1])
    number_of_concentrations = calcnum
    temperatures = list(np.full(number_of_concentrations, temperature))
    pressures = list(np.full(number_of_concentrations, 207000))

    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, 'VL', [0, 1])

    startx = x1_range[0]
    endx = x1_range[1]
    x1_linspace = np.linspace(startx, endx, number_of_concentrations)
    array1 = np.zeros((number_of_concentrations, number_of_components))
    array1[:, 0] = x1_linspace
    array1[:, 1] = 1 - x1_linspace
    array2 = np.vstack((array1, array1))
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2

    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)

    # Calculate isothemic VLE using bubble point calculation
    options['sw_Pdew'] = 0
    options['sw_Pbub'] = 1
    options['sw_Tdew'] = 0
    options['sw_Tbub'] = 0
    options['sw_PT_Flash'] = 0
    options['sw_checkAndCalcBinaryLLE'] = 0
    options['sw_lnGamma'] = 0
    options['sw_excess'] = 0

    
    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    
    #start = time.perf_counter()
    calculations_isoT = openCOSMORS.calculate(parameters, calculations, False)
    
    #end = time.perf_counter()
    #duration = end - start
    #print(f"Dauer: {duration:.6f} Sekunden")

    bp_predicted = calculations_isoT[0]['target_pressure'][:number_of_concentrations]
    bp_predicted = np.divide(bp_predicted, 10**5)

    for i in range(len(calculations)):
        number_of_components = len(calculations_isoT[i]["components"])
        concentration_array = np.ones((2*np.shape(calculations_isoT[i]["overall_concentrations"])[0],number_of_components), np.float64) 
        for j in range(np.shape(concentration_array)[0]):
            for l in range(number_of_components):
                concentration_array[j][l] = calculations_isoT[i]["concentrations"][j*number_of_components+l]
        calculations[i]["concentrations"] = concentration_array

    if len(exp_data) > 0:
        x1_data = exp_data[0]
        y1_data = exp_data[1]
        bp_data = exp_data[2]
        bp_data = [x / 10**5 for x in bp_data]
    
    # Create figure
    f = plt.figure()
    plt.xlim(0,1.0)
    legend = []
    if len(exp_data) > 0:
        plt.plot(x1_data, bp_data, "ms")
        plt.plot(y1_data, bp_data, "k^")
        legend.append("$BP$ Experiment")
        legend.append("$DP$ Experiment")

    if len(DIPPR_Psat) == 2:
        plt.plot([0, 1], [DIPPR_Psat[1], DIPPR_Psat[0]], "go")
        legend.append("DIPPR Data")

    if len(PC_Psat) == 2:
        plt.plot([0, 1], [PC_Psat[1], PC_Psat[0]], "co")
        legend.append("Pure Component Calculation")
        
    # Create plot
    plt.plot(calculations_isoT[i]["concentrations"][number_of_concentrations:, 0], bp_predicted, "r--",)
    plt.plot(calculations_isoT[i]["concentrations"][:number_of_concentrations, 0], bp_predicted, "b--",)
    legend.append("bubble-P curve")
    legend.append("dew-P curve")
    plt.legend(legend)
    plt.xlabel("$x_1, y_1$")
    plt.ylabel("Pressure / bar")
    plt.grid()
    plt.title(f"Pxy diagram at T = {temperature} K for system {comp1}/{comp2}")
    save_folder = 'VLE'
    file_name = f'VLE_isoT_{comp1}_{comp2}_{temperature}.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)

    # Create json
    BP = list(bp_predicted)
    BP = [elem[0] for elem in BP]
    result = {"Mixture": {"Component1": f"{comp1}", "Component2": f'{comp2}', "Isothermic data": {"T / K": temperatures, "P / bar": BP, "x\u2081": list(calculations_isoT[i]["concentrations"][number_of_concentrations:, 0]), "y\u2081": list(calculations_isoT[i]["concentrations"][:number_of_concentrations, 0])}}}
    file_name = f'VLE_isoT_{comp1}_{comp2}_{temperature}.json'
    full_path = os.path.join(save_folder, file_name)
    with open(full_path, 'w') as f:
        json.dump(result, f, indent=2)
    print("Calculation finished. Results saved in folder VLE.")


def VLE_isoB(components, pc_parameters, pressure, x1_range, calcnum, exp_data = []):

    '''
    Calculates the isothermic VLE for a given BINARY mixture at a given temperature.

            Parameters:
                    components (list): list of paths to COSMO files of the mixture components
                    pc_parameters (list of lists): list of lists containing the pure component parameters of the mixture components
                    pressure (float): specified pressue in Pa
                    x_range (list): range of x1 in which the VLE is calculated ([x1_low, x1_high])
                    calcnum (int): number of calculations point, example: x_range = [0.0, 1.0], calcnum = 101 -> Delta x = 0.01
                    exp_data (list of lists): possibility to provide experimental data if available, scheme: [[x_values], [y_values], [temperatures]]
                    
            Output:
                    plot and json file for given system
    '''
    
    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)

    pressure_in_bar = pressure / 1e5

    number_of_components = len(components)
    comp1 = extract_formula(components[0])
    comp2 = extract_formula(components[1])
    number_of_concentrations = calcnum
    pressures = list(np.full(number_of_concentrations, pressure))
    temperatures = []
    for i in range(number_of_concentrations):
        temperatures.append(200.0 + i * 0.01) # These temperatures are arbitrarly, but it is absolutely necessary that they differ for each concentration!!
        
    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, 'VL', [0, 1])

    startx = x1_range[0]
    endx = x1_range[1]
    x1_linspace = np.linspace(startx, endx, number_of_concentrations)
    array1 = np.zeros((number_of_concentrations, number_of_components))
    array1[:, 0] = x1_linspace
    array1[:, 1] = 1 - x1_linspace
    array2 = np.vstack((array1, array1))
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2

    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)

    # Calculate isobaric VLE using bubble point calculation
    options['sw_Pdew'] = 0
    options['sw_Pbub'] = 0
    options['sw_Tdew'] = 0
    options['sw_Tbub'] = 1
    options['sw_PT_Flash'] = 0
    options['sw_checkAndCalcBinaryLLE'] = 0
    options['sw_lnGamma'] = 0
    options['sw_excess'] = 0
    
    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    
    #start = time.perf_counter()
    calculations_isoB = openCOSMORS.calculate(parameters, calculations, False)
    
    #end = time.perf_counter()
    #duration = end - start
    #print(f"Dauer: {duration:.6f} Sekunden")

    bp_predicted = calculations_isoB[0]['temperatures'][:number_of_concentrations]

    for i in range(len(calculations)):
        number_of_components = len(calculations_isoB[i]["components"])
        concentration_array = np.ones((2*np.shape(calculations_isoB[i]["overall_concentrations"])[0],number_of_components), np.float64) 
        for j in range(np.shape(concentration_array)[0]):
            for l in range(number_of_components):
                concentration_array[j][l] = calculations_isoB[i]["concentrations"][j*number_of_components+l]
        calculations[i]["concentrations"] = concentration_array

    if len(exp_data) > 0:
        x1_data = exp_data[0]
        y1_data = exp_data[1]
        bp_data = exp_data[2]
    
    # Create figure
    f = plt.figure()
    plt.xlim(0,1.0)
    legend = []
    if len(exp_data) > 0:
        plt.plot(x1_data, bp_data, "ms")
        plt.plot(y1_data, bp_data, "k^")
        legend.append("$BP$ Experiment")
        legend.append("$DP$ Experiment")
        
    # Create plot
    plt.plot(calculations_isoB[i]["concentrations"][number_of_concentrations:, 0], bp_predicted, "r--",)
    plt.plot(calculations_isoB[i]["concentrations"][:number_of_concentrations, 0], bp_predicted, "b--",)
    legend.append("bubble-T curve")
    legend.append("dew-T curve")
    plt.legend(legend)
    plt.xlabel("$x_1, y_1$")
    plt.ylabel("Temperature / K")
    plt.grid()
    plt.title(f"Txy diagram at P = {pressure_in_bar} bar for system {comp1}/{comp2}")
    save_folder = 'VLE'
    file_name = f'VLE_isoB_{comp1}_{comp2}_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)

    # Create json
    result = {"Mixture": {"Component1": f"{comp1}", "Component2": f'{comp2}', "Isothermic data": {"T / K": list(bp_predicted), "P / bar": pressures, "x\u2081": list(calculations_isoB[i]["concentrations"][number_of_concentrations:, 0]), "y\u2081": list(calculations_isoB[i]["concentrations"][:number_of_concentrations, 0])}}}
    file_name = f'VLE_isoB_{comp1}_{comp2}_{pressure_in_bar}.json'
    full_path = os.path.join(save_folder, file_name)
    with open(full_path, 'w') as f:
        json.dump(result, f, indent=2)
    print("Calculation finished. Results saved in folder VLE.")


def calculatepTFlash_binary(components, pc_parameters, temperature, pressure, z1, calctype = 'VL', K_to_calculate = [], parameters_PT_Flash = None):
    
    '''
    Performs a pT-Flash calculation for a given BINARY mixture at a specified temperature and pressure.

            Parameters:
                    components (list): list of paths to COSMO files of the mixture components
                    pc_parameters (list of lists): list of lists containing the pure component parameters of the mixture components
                    temperature (float): specified temperature in K
                    pressure (float): specified pressure in bar
                    z1 (float): overall concentration of component 1 in binary mixture
                    calctype (str): 'VL' or 'LL'
                    K_to_calculate: default argument that allows user to provide initial K values

            Output:
                    phase compositions
    '''
    
    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)

    if parameters_PT_Flash != None:
        parameters['parameters_PT_Flash'] = parameters_PT_Flash
    
    options['sw_Pdew'] = 0
    options['sw_Pbub'] = 0
    options['sw_Tdew'] = 0
    options['sw_Tbub'] = 0
    options['sw_PT_Flash'] = 1
    options['sw_checkAndCalcBinaryLLE'] = 0
    options['sw_lnGamma'] = 0
    options['sw_excess'] = 0
    options['sw_criticalPoint'] = 0
    number_of_concentrations = 1
    number_of_components = len(components)
    temperatures = [temperature]
    pressures = [pressure]
    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, calctype, [0, 1])

    array1 = np.full((number_of_concentrations, number_of_components), [z1, 1 - z1])
    array2 = np.vstack((array1, array1))
    array3 = np.zeros((number_of_concentrations * 2, number_of_components))
    if len(K_to_calculate) > 0:
        array3 = np.full((number_of_concentrations * 2, number_of_components), K_to_calculate[:2])

    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2
    if len(K_to_calculate) > 0:
        parameters["parameters_PT_Flash"] ={
        "useGivenInitialK": True,        
        "noTracing": True                 
        }
        calculation['partition_coefficients'] = array3

    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)
    
    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)
    print(calculations)
 

def calculateActivityCoefficients(components, pc_parameters, calcnum, temperature, pressure, ln = True, exp_data = []):
    """
    Plots (logarithmic) activity coefficients (selecting liquid-like root) of all components in a (binary) mixture over the whole concentration range (concentration in mixture)

            Parameters:
                components (python list): mixture's components with paths to COSMO files
                pc_parameters (python list of python lists): list of lists with pure component EoS parameters
                calcnum (int): number of concentration points to evaluate between 0.0 and 1.0
                temperature (float): temperature for which the activity coefficients should be calculated
                pressure (float): pressure for which the activity coefficients should be calculated
                ln (bool): True = logarithmic value of AC, False = non-logarithmic value (adopted to experimental data for example)
                exp_data (python list): optionally, experimental data could be provided, format: [Source, [x_values], [AC1_values], [AC2_values]]

            Output:
                plot of activity coefficients and activities vs x
    """

    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)
    pressure_in_bar = pressure / 1e5

    number_of_components = len(components)
    comp1 = extract_formula(components[0])
    comp2 = extract_formula(components[1])
    number_of_concentrations = calcnum
    temperatures = list(np.full(number_of_concentrations, temperature))
    pressures = list(np.full(number_of_concentrations, pressure))

    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, 'LL', [0, 1])
    x1_linspace = np.linspace(0.0, 1.0, number_of_concentrations)
    array1 = np.zeros((number_of_concentrations, number_of_components))
    array1[:, 0] = x1_linspace
    array1[:, 1] = 1 - x1_linspace
    array2 = np.vstack((array1, array1))
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2
    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)
    
    options['sw_lnGamma'] = 1

    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)
    print(calculations)

    ln_AC1 = []             # list for logarithmic activity coefficients of component 1
    ln_AC2 = []             # list for logarithmic activity coefficients of component 2
    AC1 = []                # list for logarithmic activity coefficients of component 1
    AC2 = []                # list for logarithmic activity coefficients of component 2
    AC_attractive1 = []     # list for attractive part of activity coefficients of component 1
    AC_attractive2 = []     # list for attractive part of activity coefficients of component 2
    AC_repulsive1 = []      # list for repulsive part of activity coefficients of component 1
    AC_repulsive2 = []      # list for repulsive part of activity coefficients of component 2
    A1 = []                 # list for activities of component 1
    A2 = []                 # list for activities of component 2

    for i in range(number_of_concentrations):
        AC1.append(np.exp(calculations[0]['ln_phi'][i + number_of_concentrations][0] - calculations[0]['ln_phi'][2 * number_of_concentrations - 1][0]))
        AC2.append(np.exp(calculations[0]['ln_phi'][i + number_of_concentrations][1] - calculations[0]['ln_phi'][number_of_concentrations][1]))
        A1.append(x1_linspace[i] * AC1[i])
        A2.append((1-x1_linspace[i]) * AC2[i])
        if ln == True:
            ln_AC1.append(calculations[0]['ln_phi'][i + number_of_concentrations][0] - calculations[0]['ln_phi'][2 * number_of_concentrations - 1][0])
            ln_AC2.append(calculations[0]['ln_phi'][i + number_of_concentrations][1] - calculations[0]['ln_phi'][number_of_concentrations][1])
            AC_attractive1.append(calculations[0]['ln_phi_attractive'][i + number_of_concentrations][0] - calculations[0]['ln_phi_attractive'][2 * number_of_concentrations - 1][0])
            AC_attractive2.append(calculations[0]['ln_phi_attractive'][i + number_of_concentrations][1] - calculations[0]['ln_phi_attractive'][number_of_concentrations][1])
            AC_repulsive1.append(calculations[0]['ln_phi_repulsive'][i + number_of_concentrations][0] - calculations[0]['ln_phi_repulsive'][2 * number_of_concentrations - 1][0])
            AC_repulsive2.append(calculations[0]['ln_phi_repulsive'][i + number_of_concentrations][1] - calculations[0]['ln_phi_repulsive'][number_of_concentrations][1])
        

    if len(exp_data) > 1:
        source = exp_data[0]
        x1_data = exp_data[1]
        AC1_data = exp_data[2]
        AC2_data = exp_data[3]
        

    # Create figure
    f = plt.figure()
    plt.xlim(0,1.0)
    legend = []
    if len(exp_data) > 0:
        plt.plot(x1_data, AC1_data, "r")
        plt.plot(x1_data, AC2_data, "b")
        if ln == True:
            legend.append("$ln γ_1^{exp}$")
            legend.append("$ln γ_2^{exp}$")
        elif ln == False:
            legend.append("$γ_1^{exp}$")
            legend.append("$γ_2^{exp}$")
    if ln:
        plt.plot(x1_linspace, ln_AC1, "k-",)
        plt.plot(x1_linspace, ln_AC2, "k--",)
        plt.ylabel("ln $γ_i$")
        plt.title(f"ln $γ_i$ for system {comp1}/{comp2} (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
        legend.append("$ln γ_1^{calc}$")
        legend.append("$ln γ_2^{calc}$")
    else:
        plt.plot(x1_linspace, AC1, "k-",)
        plt.plot(x1_linspace, AC2, "k--",)
        plt.ylabel("$γ_i$")
        plt.title(f"$γ_i$ for system {comp1}/{comp2} (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
        legend.append("$γ_1^{calc}$")
        legend.append("$γ_2^{calc}$")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.grid()

    save_folder = 'Activities and Activity Coefficients'
    file_name = f'AC_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)
    

    f = plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, A1, "r--",)
    plt.plot(x1_linspace, A2, "b--",)
    legend.append("$a_1$ Predicted")
    legend.append("$a_2$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$a_i$")
    plt.grid()
    plt.title(f"Activities for system {comp1}/{comp2} (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'AvsX_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)

    f = plt.figure()
    plt.xlim(min(A1),max(A1))
    legend = []
    plt.plot(A1, A2, "k-",)
    plt.legend(legend)
    plt.xlabel("$a_1$")
    plt.ylabel("$a_2$")
    plt.grid()
    plt.title(f"$a_1$ vs. $a_2$, system {comp1}/{comp2} (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'A1vsA2_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)

    print("Calculation finished. Results saved in folder Activities and Activity Coefficients.")


def ExcessAndMixingPropertiesVsx(components, pc_parameters, x_range, calcnum, temperature, pressure, parameters_mixing = [], exp_data = []):
    """
    Plots excess and mixing properties of a (binary) mixture over the whole concentration range (concentration in mixture)

            Parameters:
                components (python list): mixture's components with paths to COSMO files
                pc_parameters (python list of python lists): list of lists with pure component EoS parameters
                x_range (list): range of x1 in which the excess and mixing  properties are calculated ([x1_low, x1_high])
                calcnum (int): number of concentration points to evaluate between 0.0 and 1.0
                temperature (float): temperature for which the excess / mixing properties should be calculated
                pressure (float): pressure for which the excess / mixing properties should be calculated
                exp_data (bool): True = experimental data are availabe and are plotted in addition to the calculated values

            Output:
                Plots and json with the results for given system
    """

    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)

    if len(parameters_mixing) > 0:
        parameters['parameters_mixing'] = parameters_mixing
        
    pressure_in_bar = pressure / 1e5
    startx = x_range[0]
    endx = x_range[1]

    number_of_components = len(components)
    comp1 = extract_formula(components[0])
    comp2 = extract_formula(components[1])
    number_of_concentrations = calcnum
    temperatures = list(np.full(number_of_concentrations, temperature))
    pressures = list(np.full(number_of_concentrations, pressure))

    calculation = addCalculationToStructure(components, number_of_concentrations, temperatures, pressures, 'VL', [0, 1])
    
    x1_linspace = np.linspace(startx, endx, number_of_concentrations)
    array1 = np.zeros((number_of_concentrations, number_of_components))
    array1[:, 0] = x1_linspace
    array1[:, 1] = 1 - x1_linspace
    array2 = np.vstack((array1, array1))
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2
    
    calculations = []
    calculations.append(calculation)
    
    options['sw_excess'] = 1
    parameters['exp'] = {'NumericalDerivative': True, 'EigenMatrixToFile': 1}

    calculations = fill_missing_calculation_structures(calculations, options)
    
    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)
    
    gE = []
    gM = []
    hE = []
    cpE = []
    hM = []
    cpM = []

    for i in range(number_of_concentrations):
        gE.append(calculations[0]['gE'][i][0] + calculations[0]['gE'][i + number_of_concentrations][0])
        hE.append(calculations[0]['hE'][i][0] + calculations[0]['hE'][i + number_of_concentrations][0])
        cpE.append(calculations[0]['cpE'][i][0] + calculations[0]['cpE'][i + number_of_concentrations][0])
        gM.append(calculations[0]['gM'][i][0] + calculations[0]['gM'][i + number_of_concentrations][0])
        hM.append(calculations[0]['hM'][i][0] + calculations[0]['hM'][i + number_of_concentrations][0])
        cpM.append(calculations[0]['cpM'][i][0] + calculations[0]['cpM'][i + number_of_concentrations][0])
        

    # Create figure
    f = plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, gE, "k--",)
    legend.append("$g^E$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$g^E [J/mol]$")
    plt.grid()
    plt.title(f"Plot of $g^E$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    save_folder = 'Excess and mixing properties'
    file_name = f'gE_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(full_path)
    
    plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, gM, "k--",)
    legend.append("$g^M$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$g^M [J/mol]$")
    plt.grid()
    plt.title(f"Plot of $g^M$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'gM_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    plt.savefig(full_path)

    plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, hE, "k--",)
    legend.append("$h^E$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$h^E [J/mol]$")
    plt.grid()
    plt.title(f"Plot of $h^E$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'hE_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    plt.savefig(full_path)

    plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, hM, "k--",)
    legend.append("$h^M$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$h^M [J/mol]$")
    plt.grid()
    plt.title(f"Plot of $h^M$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'hM_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    plt.savefig(full_path)
    
    plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, cpE, "k--",)
    legend.append("$c_p^E$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$c_p^E [J/mol/K]$")
    plt.grid()
    plt.title(f"Plot of $c_p^E$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'cpE_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    plt.savefig(full_path)

    plt.figure()
    plt.xlim(0,1.0)
    legend = []
    plt.plot(x1_linspace, cpM, "k--",)
    legend.append("$c_p^M$ Predicted")
    plt.legend(legend)
    plt.xlabel("$x_1$")
    plt.ylabel("$c_p^M [J/mol/K]$")
    plt.grid()
    plt.title(f"Plot of $c_p^M$ for {comp1}/{comp2} against mole fraction at (T = {temperature} K, P = {pressure_in_bar} bar)", fontsize=10)
    file_name = f'cpM_{comp1}_{comp2}_{temperature}_K_{pressure_in_bar}_bar.png'
    full_path = os.path.join(save_folder, file_name)
    plt.savefig(full_path)

    # Create json
    x1 = list(x1_linspace)
    result = {"Mixture": {"Component1": f"{comp1}", "Component2": f'{comp2}', "T / K": temperature, "P / bar": pressure_in_bar, "z\u2081": x1, "gE": gE, "gM": gM, "hE": hE, "hM": hM, "cpE": cpE, "cpM": cpM}}
    file_name = f'ExcessMixing_{comp1}_{comp2}_{temperature}.json'
    full_path = os.path.join(save_folder, file_name)
    with open(full_path, 'w') as f:
        json.dump(result, f, indent=2)

    options['sw_excess'] = 0
    print("Calculation finished. Results saved in folder Excess and mixing properties.")



# Test function calls

# Provide paths to COSMO files in list 'components'
# Provide EoS Parameters for pure components in list 'pc_parameters' with format: [[bi_comp1, delta_m0_comp1, delta_mT_comp1, bh_comp1], [bi_comp2, delta_m0_comp2, delta_mT_comp2, bh_comp2], ...]

try:
    ##### ISOTHERMIC VLE #####
    #components = [
    #    r"cyclohexane.orcacosmo",
    #    r"methylcyclohexane.orcacosmo"
    #]
    #pc_parameters = [[153.60645445740928, 1.0336287205332177, 4.423459633632872, 16.979264160185735], [182.0802815397086, 0.970174606987601, 4.236619184631015, 18.484240761367914]]
    #temperature = 308.15 # in K
    #VLE_isoT(components, pc_parameters, temperature, [0.0, 1.0], 101)


    ##### ISOBARIC VLE #####
    #components = [
    #    r"cyclohexane.orcacosmo",
    #    r"methylcyclohexane.orcacosmo"
    #]
    #pc_parameters = [[153.60645445740928, 1.0336287205332177, 4.423459633632872, 16.979264160185735], [182.0802815397086, 0.970174606987601, 4.236619184631015, 18.484240761367914]]
    #pressure = 101300.0 # in Pa
    #VLE_isoB(components, pc_parameters, pressure, [0.0, 1.0], 101)


    ##### CALCULATE A DEW PRESSURE #####
    #components = [
    #    r"cyclohexane.orcacosmo",
    #    r"methylcyclohexane.orcacosmo"
    #]
    #pc_parameters = [[153.60645445740928, 1.0336287205332177, 4.423459633632872, 16.979264160185735], [182.0802815397086, 0.970174606987601, 4.236619184631015, 18.484240761367914]]
    #temperature = 308.15 # in K
    #x1_value = 0.4
    #parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)
    #temperatures = [308.15]
    #pressures = [207000] # random value
    #calculation = addCalculationToStructure(components, 1, temperatures, pressures, 'VL', [0, 1])
    #array1 = np.array([x1_value, 1-x1_value])
    #array2 = np.vstack((array1, array1))
    #calculation['overall_concentrations'] = array1
    #calculation['concentrations'] = array2

    #calculations = []
    #calculations.append(calculation)
    #calculations = fill_missing_calculation_structures(calculations, options)
    #options['sw_Pdew'] = 1
    #openCOSMORS.initialize()
    #openCOSMORS.loadMolecules(options, parameters, components)
    #openCOSMORS.loadCalculations(calculations)
    #calculations = openCOSMORS.calculate(parameters, calculations, False)
    
    #print(f"The dew pressure for x1 = {x1_value} is {calculations[0]['total_pressure'][1][0]} Pa.")



    ##### VLE PT FLASH CALCULATION FOR BINARY MIXTURE AT SPECIFIED P, T #####
    #components = [
    #    r"cyclohexane.orcacosmo",
    #    r"methylcyclohexane.orcacosmo"
    #]
    #pc_parameters = [[153.60645445740928, 1.0336287205332177, 4.423459633632872, 16.979264160185735], [182.0802815397086, 0.970174606987601, 4.236619184631015, 18.484240761367914]]
    #temperature = 308.15 # in K
    #pressure = 14000 # in Pa
    #z1 = 0.5 # overall mixture concentration of component 1

    # Hyperparameters for some algorithms to calculate thermodynamic properties
    
    # Optional Hyperparameters for pT Flash
    #parameters_PT_Flash = {
    #    "useGivenInitialK": False,         # not specified or False: pT-Flash estimates initial K values based on Raoult's law. 
                                           # True: pT-Flash uses K values provided by the model user as initial estimates.
    #    "noTracing": True                 # If initial K values do not lead to convergence the algorithm starts a Bubble P tracing procedure
                                           # to find good K values. By setting this to 'True' this procedure is omitted.
    #   }

    #calculatepTFlash_binary(components, pc_parameters, temperature, pressure, z1, 'VL', K_to_calculate = [], parameters_PT_Flash = None)


    ##### ACTIVITY COEFFICIENTS #####
    components = [
        r'nButane.orcacosmo',
        r'methanol.orcacosmo'
       ]
    pc_parameters = [[126.3051060272007, 0.8597839041393577, 3.656973366271159, 15.658181790973256], [56.08186814344738, 1.3518314651065213, 3.3225493224786162, 8.713083189565843]]
    calcnum = 101
    temperature = 273
    pressure = 101325
    calculateActivityCoefficients(components, pc_parameters, calcnum, temperature, pressure, ln = True, exp_data = [])



    ##### EXCESS AND MIXING PROPERTIES #####
    #components = [
    #     r'nhexane..orcacosmo',
    #     r'cyclohexane.orcacosmo'
    #    ]
    #pc_parameters = [[180.66276625980947, 0.9146656325217001, 3.3666431129786405, 17.692418203607357], [153.60645445740928, 1.0336287205332177, 4.423459633632872, 16.979264160185735]]
    #temperature = 413.21
    #pressure = 1134000

    # Optional Hyperparameters for mixing properties
    # Normally, the implementation for mixing and excess properties first checks for a possible phase split.
    # However, if the phase boundaries for given p,T are already known you can provide them using this scheme.
    # You can also use this scheme to skip the Flash procedure if you know that for specified p,T only 1 phase occurs. For this set all concentrations to 0 if the mixture is a single liquid phase and all to 1 if mixture is a single gaseous phase.
    #parameters_mixing = {
    #    "temperatures": [413.21],
    #    "pressures": [1134000],
    #    "concentration_x_phase": [[0.0, 0.0]],
    #    "concentration_y_phase": [[0.0, 0.0]],
    #}
    #ExcessAndMixingPropertiesVsx(components, pc_parameters, [0.0, 1.0], 101, temperature, pressure, parameters_mixing, [])


    ##### CRITICAL POINT #####
    # Warning: Time consuming
    #components = [
    #     r'methane.orcacosmo',
    #     r'ethane.orcacosmo'
    #    ]
    #pc_parameters = [[50.826951478727125, 0.5943262353122482, 3.1858544427725564, 10.669150253967132], [73.48869746828207, 0.8031718195904861, 3.548049119383619, 12.683442985210894]]
    #temperature = 210
    #calculateCriticalOrAzeotropicPoint(components, pc_parameters, temperature, 'critical')


except Exception as err:
    print('ERROR:')
    print(type(err))
    print(err.args)
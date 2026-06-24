
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
    "sw_residual_properties": 1,

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


# Provide paths to COSMO files in list 'components'
# Provide EoS Parameters for pure components in list 'pc_parameters' with format: [[bi_comp1, delta_m0_comp1, delta_mT_comp1, bh_comp1], [bi_comp2, delta_m0_comp2, delta_mT_comp2, bh_comp2], ...]

try:
    ### EXAMPLE: CALCULATION OF VAPOR PRESSURE OF METHANOL FOR A SINGLE TEMPERATURE ###
    ### USER INPUT HERE ###
    components = [
        r'methanol.orcacosmo'
        ]
    pc_parameters = [[56.08186814344738, 1.3518314651065213, 3.3225493224786162, 8.713083189565843]]
    temperature = input("Enter temperature for vapor pressure calculation of methanol in K: ")
    pressure = 101325 # random value
    
    
    parameters['pure_component_parameters'] = np.array(pc_parameters).astype(np.float64)
    temperatures = list(np.full(1, temperature))
    pressures = list(np.full(1, pressure))
    calculation = addCalculationToStructure(components, 1, temperatures, pressures, 'LL', [0])
    array1 = np.array([1.0])
    array2 = np.vstack((array1, array1))
    calculation['overall_concentrations'] = array1
    calculation['concentrations'] = array2
    calculations = []
    calculations.append(calculation)
    calculations = fill_missing_calculation_structures(calculations, options)

    openCOSMORS.initialize()
    openCOSMORS.loadMolecules(options, parameters, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(parameters, calculations, False)
    
    pvap = calculations[0]["total_pressure"][1][0]
    print(f"The vapor pressure of methanol at {temperature} K is {pvap} Pa.")


except Exception as err:
    print('ERROR:')
    print(type(err))
    print(err.args)
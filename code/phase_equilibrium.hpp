#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <limits>

#include "types.hpp"

// ====================== FUNCTION DECLARATIONS ======================
void preparePhaseEquilibriumCalculations(parameters& param, calculation& calc);
void performPhaseEquilibriumCalculations(parameters& param, calculation& calc);

void updateBubblePressure(calculation& calc, int i_calculation, int numberOfCalculations, double sum_yt, double& lowerP, double& lowerSum, double& upperP, double& upperSum);
void updateBubbleTemperature(parameters& param, calculation& calc, int i_calculation, int numberOfCalculations, double& sum_yt, double& last_sum, int& outlier_counter, double& lowerT, double& lowerSum, double& upperT, double& upperSum);
void recoverBubblePressureBySampling(parameters& param, calculation& calc, int i_calculation, std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y, std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_P);
void recoverBubbleTemperatureBySampling(parameters& param, calculation& calc, int i_calculation, std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y, std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_T);
void calculateBubblePoint(calculation& calc, int i_calculation, double epsilon = 0.00001, bool passInnerLoop = false);
void calculateBubblePoints(parameters& param, calculation& calc, double tolerance = 0.00001);

void updateDewPressure(calculation& calc, int i_calculation, int numberOfCalculations, double sum_xt, double& lowerP, double& lowerSum, double& upperP, double& upperSum);
void updateDewTemperature(parameters& param, calculation& calc, int i_calculation, int numberOfCalculations, double& sum_xt, double& last_sum, int& outlier_counter, double& lowerT, double& lowerSum, double& upperT, double& upperSum);
void calculateDewPoint(calculation& calc, int i_calculation, double epsilon = 0.00001);
void calculateDewPoints(parameters& param, calculation& calc, double tolerance = 0.00001);

double initializeLLEFlash(parameters& param, calculation& calc, int i_calculation, double& thetaGuess, bool& skipThetaCalculation);
void initializeAndSolveVLEFlashBySampling(parameters& param, calculation& calc, int i_calculation, std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y, std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_P);
void calculatePTFlash(calculation calc, int i_calculation, double ConvergenceCriteria, bool skipThetaCalculation, double thetaGuess = 0.0, bool UseGivenK = false);
void runPTFlash(parameters& param, calculation& calc);

void findCriticalPointForSpecifiedx(calculation& calc, int i_calculation);
void calculateCriticalPointForSpecifiedx(parameters& param, calculation& calc);
void calculateCriticalOrAzeotropicPointForSpecifiedT(parameters& param, calculation& calc);

void computeMixingForPhase(calculation& calc, int i_calculation, int phaseIndex, int numberOfOriginalComponents, double temperature, double weight);
void computeExcessForPhase(calculation& calc, int i_calculation, int phaseIndex, int numberOfOriginalComponents, double temperature, double weight);
void calculateExcessAndMixingProperties(parameters& param, calculation& calc);

void computeActivityCoefficients(parameters& param, calculation& calc);

void calculateSubcalculations(std::vector<int>& calculationIndices);


// ====================== MAIN FUNCTION ======================
/**
 * @brief Starts the individual functions to calculate the phase equilibrium property specified by user
 * via switch (sw_Tdew, sw_Tbub, sw_Pdew, sw_Pbub, sw_PT_Flash, sw_criticalPoint, sw_azeotropicPoint).
 * Noe: Currently, only one of the switches above may be active for a calculations vector.
 */
inline void performPhaseEquilibriumCalculations(parameters& param, calculation& calc)
{
    preparePhaseEquilibriumCalculations(param, calc);
    if (param.sw_Tbub == 1 || param.sw_Pbub == 1)
    {
        calculateBubblePoints(param, calc);
    }
    else if (param.sw_Pdew == 1 || param.sw_Tdew == 1) {
        calculateDewPoints(param, calc);
    }
    else if (param.sw_PT_Flash == 1) {
        runPTFlash(param, calc);
    }
    else if (param.sw_criticalPoint_specifiedx == 1) {
        calculateCriticalPointForSpecifiedx(param, calc);
    }
    else if (param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) {
        calculateCriticalOrAzeotropicPointForSpecifiedT(param, calc);
    }
    else if (param.sw_excess == 1) {
        calculateExcessAndMixingProperties(param, calc);
    }
    else if (param.sw_lnGamma == 1) {
        computeActivityCoefficients(param, calc);
    }
}



/**
 * @brief Prepare all quantities required for subsequent phase equilibrium calculations.
 *
 * The routine is active only for vapor–liquid calculations or when an excess
 * or mixing properties are to be calculated (calc.calculationType == "VL" or param.sw_excess == 1).
 * Depending on the active switches (sw_Tdew, sw_Tbub, sw_Pdew, sw_Pbub, sw_PT_Flash,
 * sw_criticalPoint, sw_azeotropicPoint), it prepares the required saturation
 * properties and consistent initial guesses for T, P, and K.
 */
inline void preparePhaseEquilibriumCalculations(parameters& param, calculation& calc) {

    int numberOfOriginalComponents = int(calc.components.size()) - 1;
    calc.overallConcentrations = calc.originalConcentrations.cast<double>();
    calc.PartitionCoefficients = Eigen::MatrixXd(int(calc.originalConcentrations.rows()), numberOfOriginalComponents);

    if (calc.calculationType == "VL" || param.sw_excess == 1) {

        std::vector<calculation> IntermediateSubcalculations;

        if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
            calc.SaturationTemperatures = Eigen::MatrixXd(int(calc.originalConcentrations.rows()), numberOfOriginalComponents);

            // Find pure component saturation temperature previously calculated for specified P
            for (int i_pressure = 0; i_pressure < int(calc.originalConcentrations.rows()); i_pressure++) {
                double pressureToFind = calc.targetPressure(i_pressure);
                int index = -1;
                for (int k = 0; k < int(calc.uniquePressures.size()); k++) {
                    if (calc.uniquePressures[k] == pressureToFind) {
                        index = k;
                        break;
                    }
                }
                for (int l = 0; l < calc.components.size() - 1; l++) {
                    int subcalcindex = calc.IndicesForPureComponentCalculations(index, l);
                    calc.SaturationTemperatures(i_pressure, l) = subcalculations[subcalcindex].temperatures[0];
                }
            }

            // Initial estimates for boiling temperature
            Eigen::VectorXd target_temperatures = (calc.overallConcentrations * calc.SaturationTemperatures.transpose()).diagonal();

            // Save initial boiling temperature estimate in temperatures and TauTemperatures
            for (int i_pressure = 0; i_pressure < int(calc.originalConcentrations.rows()); i_pressure++) {
                int actualConcentrationIndex = calc.actualConcentrationIndices[i_pressure];
                int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_pressure][0]];
                calc.temperatures[actualConcentrationIndex] = target_temperatures(i_pressure);
                calc.temperatures[actualReferenceStateConcentrationIndex] = target_temperatures(i_pressure);
                for (int g = 0; g < calc.TauConcentrationIndices.size(); g++) {
                    for (int h = 0; h < calc.TauConcentrationIndices[g].size(); h++) {
                        if (actualConcentrationIndex == calc.TauConcentrationIndices[g][h]) {
                            calc.TauTemperatures[g] = target_temperatures(i_pressure);
                        }
                    }
                }
            }

            // Now where we have first guesses for the temperatures, we calculate the saturation pressures at these temperatures
            IntermediateSubcalculations = subcalculations;
            subcalculations.clear();
            subcalculations.reserve(10000);
            for (int k = 0; k < target_temperatures.size(); k++) {
                calc.uniqueTemperatures.push_back(target_temperatures(k));
            }
            std::sort(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end());
            calc.uniqueTemperatures.erase(std::unique(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end()), calc.uniqueTemperatures.end());
            calc.IndicesForPureComponentCalculations = Eigen::MatrixXi(int(calc.uniqueTemperatures.size()), calc.components.size() - 1);
            for (int j = 0; j < calc.components.size() - 1; j++) {
                std::vector<std::shared_ptr<molecule>> actualMolecules;
                actualMolecules.push_back(calc.components[j]);
                actualMolecules.push_back(calc.components[int(calc.components.size() - 1)]);
                int subcalculationIndex = -2;
                for (int k = 0; k < calc.uniqueTemperatures.size(); k++) {
                    std::vector<double> temperatureVector;
                    temperatureVector.push_back(calc.uniqueTemperatures[k]);
                    subcalculationIndex = calc.addOrFindArrayIndexInSubcalculations(subcalculations, actualMolecules, calc.uniqueTemperatures[k]);
                    if (subcalculationIndex == -1) {
                        subcalculations.push_back(loadSubcalculation(actualMolecules, temperatureVector, { {1.0, 0.0} }));
                        subcalculationIndex = int(subcalculations.size()) - 1;
                    }
                    calc.IndicesForPureComponentCalculations(k, j) = subcalculationIndex;
                }
            }
            param.sw_Pbub = 1;
            const size_t numSubCalcs = subcalculations.size();
            std::vector<int> subcalculationIndices(numSubCalcs);
            for (int i = 0; i < numSubCalcs; i++) {
                subcalculationIndices[i] = i;
            }
            calculateSubcalculations(subcalculationIndices);
            param.sw_Pbub = 0;
        }

        if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_Tdew == 1 || param.sw_Tbub == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_excess == 1) {
            calc.SaturationPressures = Eigen::MatrixXd(int(calc.originalConcentrations.rows()), numberOfOriginalComponents);

            // Find pure component saturation pressure previously calculated for specified T
            for (int i_temperature = 0; i_temperature < int(calc.originalConcentrations.rows()); i_temperature++) {
                int actualConcentrationIndex = calc.actualConcentrationIndices[i_temperature];
                double temperatureToFind = calc.temperatures[actualConcentrationIndex];
                auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperatureToFind);
                int index = it - calc.uniqueTemperatures.begin();
                for (int l = 0; l < calc.components.size() - 1; l++) {
                    int subcalcindex = calc.IndicesForPureComponentCalculations(index, l);
                    calc.SaturationPressures(i_temperature, l) = subcalculations[subcalcindex].SaturationPressures(0, 0); // look
                }
            }

            // In case of a Bubble P, Dew P, Critical Point or Azeotropic Point Calculation, total pressure P is unknown and therefore estimated using Raoult's law
            if (param.sw_Pbub == 1 || param.sw_Pdew == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) {
                Eigen::VectorXd TargetPressureAccordingToRaoult = (calc.overallConcentrations * calc.SaturationPressures.transpose()).diagonal();
                calc.targetPressure = TargetPressureAccordingToRaoult;
            }

            // Calculation of initial K values
            for (int k = 0; k < calc.originalConcentrations.rows(); k++) {
                for (int l = 0; l < calc.components.size() - 1; l++) {
                    calc.PartitionCoefficients(k, l) = calc.SaturationPressures(k, l) / calc.targetPressure(k);
                }
            }

            if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
                subcalculations = IntermediateSubcalculations;
            }
        }
    }
}


// ====================== HELPER FUNCTIONS ======================
/**
 * @brief Result of binary component classification into supercritical/subcritical.
 */
struct ComponentClassification {
    int supercriticalIndex;          // 0 or 1: which component is "supercritical"
    int subcriticalIndex;            // the other one
    int supercriticalSubcalcIndex;   // index into subcalculations[] for the supercritical component
    int subcriticalSubcalcIndex;     // index into subcalculations[] for the subcritical component
};

/**
 * @brief Classifies two binary components as supercritical/subcritical relative to a reference value.
 *
 * For Bubble P / PT-Flash: compare critical temperatures against current temperature.
 * For Bubble T: compare critical pressures against current pressure.
 *
 * Logic:
 * - If comp1 is above critical and comp2 is below: comp1 is supercritical
 * - If comp2 is above critical and comp1 is below: comp2 is supercritical
 * - If both are below critical (e.g. both Tc > T): comp2 is chosen as "supercritical"
 *   (convention: higher-index component is used as tracing direction)
 * - If both are above critical: error (currently no implementation)
 *
 * @param subcalcindex_comp1: index of component 1 in subcalculations[]
 * @param subcalcindex_comp2: index of component 2 in subcalculations[]
 * @param criticalValue1: critical T (or P) of component 1
 * @param criticalValue2: critical T (or P) of component 2
 * @param referenceValue: current temperature (or pressure) to compare against
 * @return ComponentClassification with indices set
 * @throws std::runtime_error if both components are in supercritical state
 */
inline ComponentClassification classifyComponents(int subcalcindex_comp1, int subcalcindex_comp2, double criticalValue1, double criticalValue2, double referenceValue, bool raiseError = true)
{
    ComponentClassification result;

    bool comp1_supercritical = (criticalValue1 < referenceValue);  // Tc < T means supercritical
    bool comp2_supercritical = (criticalValue2 < referenceValue);

    if (comp1_supercritical && !comp2_supercritical) {
        // Component 1 is supercritical, component 2 is subcritical
        result.supercriticalIndex = 0;
        result.supercriticalSubcalcIndex = subcalcindex_comp1;
        result.subcriticalIndex = 1;
        result.subcriticalSubcalcIndex = subcalcindex_comp2;
    }
    else if (!comp1_supercritical && comp2_supercritical) {
        // Component 2 is supercritical, component 1 is subcritical
        result.supercriticalIndex = 1;
        result.supercriticalSubcalcIndex = subcalcindex_comp2;
        result.subcriticalIndex = 0;
        result.subcriticalSubcalcIndex = subcalcindex_comp1;
    }
    else if (!comp1_supercritical && !comp2_supercritical) {
        // Both subcritical (both Tc > T): convention — use comp2 as tracing direction
        result.supercriticalIndex = 1;
        result.supercriticalSubcalcIndex = subcalcindex_comp2;
        result.subcriticalIndex = 0;
        result.subcriticalSubcalcIndex = subcalcindex_comp1;
    }
    else {
        // Both supercritical
        if (raiseError) {
            throw std::runtime_error(
                "Both components are in supercritical state. Sampling currently doesn't work for this situation.");
        }
        else {
            // This is just to indicate that both components are supercritical
            result.supercriticalIndex = 1;
            result.supercriticalSubcalcIndex = 1;
            result.subcriticalIndex = 1;
            result.subcriticalSubcalcIndex = 1;
        }
    }

    return result;
}


// ====================== FUNCTIONS NEEDED IN CONTEXT OF SAMPLING ======================
/**
 * @brief Determines the concentration and pressure ranges covered by sampling data.
 */
struct SamplingRange {
    double min_conc = 1.0;
    double max_conc = 0.0;
    double min_P = 0.0;
    double max_P = 0.0;
};

inline SamplingRange determineSamplingRange(
    const std::vector<Eigen::VectorXd>& sampling_x,
    const std::vector<Eigen::VectorXd>& sampling_y,
    const std::vector<Eigen::VectorXd>& sampling_P,
    int supercriticalIndex)
{
    SamplingRange range;
    range.min_P = sampling_P[0](0);
    range.max_P = sampling_P[0](0);

    for (size_t i = 0; i < sampling_P.size(); i++) {
        double P = sampling_P[i](0);
        double x_sc = sampling_x[i](supercriticalIndex);
        double y_sc = sampling_y[i](supercriticalIndex);

        if (P < range.min_P) range.min_P = P;
        if (P > range.max_P) range.max_P = P;

        double local_min = std::min(x_sc, y_sc);
        double local_max = std::max(x_sc, y_sc);

        if (local_min < range.min_conc) range.min_conc = local_min;
        if (local_max > range.max_conc) range.max_conc = local_max;
    }

    return range;
}


/**
 * @brief Result of the two-phase segment search within sampling data.
 */
struct SegmentSearchResult {
    bool twoPhaseFound = false;
    bool zInAnySegment = false;
    int twoPhaseIndex = -1;
    int segmentIndex = -1;
    double P_low = 0.0;   // pressure bounds of the segment containing z
    double P_high = 0.0;
};

/**
 * @brief Searches sampling data to determine whether a given (z, P) point falls
 *        within the two-phase region.
 *
 * The function iterates over consecutive sampling segments and checks:
 * 1. Whether the overall composition z falls within the concentration envelope
 *    (min(x,y) to max(x,y)) of a segment
 * 2. Whether the target pressure falls within the pressure range of that segment
 *
 * @param sampling_x Liquid phase compositions from sampling
 * @param sampling_y Vapor phase compositions from sampling
 * @param sampling_P Pressures from sampling
 * @param supercriticalIndex Component index used for comparison (0 or 1)
 * @param z Overall composition of the supercritical component
 * @param P_target Target pressure
 * @param range Pre-computed sampling range (from determineSamplingRange)
 * @return SegmentSearchResult with search outcome
 */
inline SegmentSearchResult findTwoPhaseSegment(
    const std::vector<Eigen::VectorXd>& sampling_x,
    const std::vector<Eigen::VectorXd>& sampling_y,
    const std::vector<Eigen::VectorXd>& sampling_P,
    int supercriticalIndex,
    double z,
    double P_target,
    const SamplingRange& range)
{
    SegmentSearchResult result;

    // Quick check: is z within the overall sampling range?
    if (sampling_P.size() <= 1 || z < range.min_conc || z > range.max_conc) {
        return result;
    }

    for (size_t i = 1; i < sampling_P.size(); ++i) {
        double xmin_prev = std::min(sampling_x[i - 1](supercriticalIndex),
            sampling_y[i - 1](supercriticalIndex));
        double xmax_prev = std::max(sampling_x[i - 1](supercriticalIndex),
            sampling_y[i - 1](supercriticalIndex));

        double xmin_curr = std::min(sampling_x[i](supercriticalIndex),
            sampling_y[i](supercriticalIndex));
        double xmax_curr = std::max(sampling_x[i](supercriticalIndex),
            sampling_y[i](supercriticalIndex));

        double zmin = std::min(xmin_prev, xmin_curr);
        double zmax = std::max(xmax_prev, xmax_curr);

        // Check if z falls in this segment's concentration envelope
        if (!(zmin <= z && z <= zmax)) {
            continue;
        }

        result.zInAnySegment = true;
        result.segmentIndex = static_cast<int>(i);

        double P_prev = sampling_P[i - 1](0);
        double P_curr = sampling_P[i](0);

        result.P_low = std::min(P_prev, P_curr);
        result.P_high = std::max(P_prev, P_curr);

        // Check if target pressure falls within this segment's pressure range
        if (result.P_low <= P_target && P_target <= result.P_high) {
            double xmin_seg = std::min(xmin_prev, xmin_curr);
            double xmax_seg = std::max(xmax_prev, xmax_curr);

            if (xmin_seg <= z && z <= xmax_seg) {
                result.twoPhaseFound = true;
                result.twoPhaseIndex = static_cast<int>(i) - 1;
                break;
            }
        }
    }

    return result;
}


std::vector<std::vector<Eigen::VectorXd>> samplingForVLE(std::vector<std::shared_ptr<molecule>> components, double temperature, int IndexOfSupercriticalComponent, Eigen::VectorXd startx, Eigen::VectorXd starty, Eigen::VectorXd startK, double startpressure, std::vector<double> PureTsat = {}, double end_x = 1.0) {

    std::vector<std::vector<double>> concentrationx1(1);
    int IndexOfSubcriticalComponent;
    if (IndexOfSupercriticalComponent == 0) {
        IndexOfSubcriticalComponent = 1;
        concentrationx1[0].push_back(startx(0) + 0.0001);
        concentrationx1[0].push_back(startx(1) - 0.0001);
    }
    if (IndexOfSupercriticalComponent == 1) {
        IndexOfSubcriticalComponent = 0;
        concentrationx1[0].push_back(startx(0) - 0.0001);
        concentrationx1[0].push_back(startx(1) + 0.0001);
    }

    std::vector<Eigen::VectorXd> x = {};
    std::vector<Eigen::VectorXd> y = {};
    std::vector<Eigen::VectorXd> K = {};
    std::vector<Eigen::VectorXd> Pressures = {};
    std::vector<Eigen::VectorXd> Temperatures = {};

    x.push_back(startx);
    K.push_back(startK);
    y.push_back(starty);
    Eigen::VectorXd pressureVec(1);
    pressureVec(0) = startpressure;
    Pressures.push_back(pressureVec);
    Eigen::VectorXd tempVec(1);
    tempVec(0) = temperature;
    Temperatures.push_back(tempVec);

    std::vector<double> pressureVector;
    pressureVector.push_back(double(startpressure));

    double DeltaT = 0.0;
    if (PureTsat.size() > 0) {
        DeltaT = 0.001 * (PureTsat[IndexOfSupercriticalComponent] - PureTsat[IndexOfSubcriticalComponent]);
    }

    // To not affect the actual concentrations given exernally, the tracing procedure uses a new calculaion structure that is here created first
    calculation calculationScheme = loadSubcalculation(components, { double(temperature + DeltaT) }, concentrationx1, pressureVector);
    int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
    calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});
    calculationScheme.PartitionCoefficients = Eigen::MatrixXd::Zero(2, 2);
    calculationScheme.PartitionCoefficients.row(0) = startK;
    for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
        double value = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
        calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
    }
    calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
    int numberOfOriginalComponents = calculationScheme.components.size() - 1;
    calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.calculationType = "VL";
    calculationScheme.originalNumberOfCalculations = 2;
    for (int i = 0; i < numberOfOriginalComponents; i++) {
        int index = calculationScheme.components[i]->index;
        calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
        calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
    }


    bool VLEDoesNotConvergeAnymore = false;
    int i_calculation = 0;
    int actualsize = 0;
    double stepsize = 0.0001;
    double last_x = 0.0;
    int counter_for_x_almost_y = 0;
    while (VLEDoesNotConvergeAnymore != true && last_x <= end_x) {
        calculateBubblePoint(calculationScheme, i_calculation, 0.00001);
        if (param.sw_Pbub == 1 && param.sw_criticalPoint != 1 && param.sw_azeotropicPoint != 1 && calculationScheme.totalPressure(i_calculation) == 0.0) {
            VLEDoesNotConvergeAnymore = true;
            break;
        }
        else if (param.sw_Pbub == 1 && (param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) && calculationScheme.totalPressure(i_calculation) == 0.0) {
            // Try 10 smaller stepsizes
            int counter = 1;
            while (calculationScheme.totalPressure(i_calculation) == 0.0) {
                Eigen::VectorXd newxconcentration(2);
                int index = std::max(0, actualsize - 1);
                newxconcentration(IndexOfSupercriticalComponent) = std::min(x[index](IndexOfSupercriticalComponent) + 1 / std::pow(2, counter) * stepsize, 1.0);
                newxconcentration(IndexOfSubcriticalComponent) = std::max(x[index](IndexOfSubcriticalComponent) - 1 / std::pow(2, counter) * stepsize, 0.0);
                calculationScheme.originalConcentrations.row(1) = newxconcentration;
                for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                    double value = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
                    calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
                }
                calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
                calculateBubblePoint(calculationScheme, i_calculation, 0.00001);
                counter += 1;
                if (counter > 10) {
                    break;
                }
            }
            if (calculationScheme.totalPressure(i_calculation) == 0.0 || counter > 10) {
                // No convergence (VLE) for the specified x, T anymore
                VLEDoesNotConvergeAnymore = true;
                break;
            }
        }
        else if (param.sw_Tbub == 1 && calculationScheme.targetTemperature(i_calculation) == 0.0) {
            VLEDoesNotConvergeAnymore = true;
            break;
        }
        if ((param.sw_Pbub == 1 && calculationScheme.totalPressure(i_calculation) != 0.0) || (param.sw_Tbub == 1 && calculationScheme.targetTemperature(i_calculation) != 0.0)) {

            Pressures.push_back(calculationScheme.totalPressure.row(1));
            Eigen::VectorXd tempVec(1);
            tempVec(0) = calculationScheme.targetTemperature(i_calculation);
            Temperatures.push_back(tempVec);
            x.push_back(calculationScheme.originalConcentrations.row(1));
            last_x = calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent);

            y.push_back(calculationScheme.originalConcentrations.row(0));
            calculationScheme.PartitionCoefficients.row(0) = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();
            K.push_back(calculationScheme.PartitionCoefficients.row(0));

            if (int(x.size()) > 1 && (calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) == 0.0 || calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) == 1.0)) {
                break;
            }

            if ((param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) && 
                abs(calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) - calculationScheme.originalConcentrations(0, IndexOfSupercriticalComponent)) < 0.0001 && 
                calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) > 0.02) {

                // Compositions are close — determine whether this is an azeotrope or critical point

                calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
                calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;
                param.sw_isotherm = 1;
                std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 1);
                param.sw_isotherm = 0;
                std::vector<double> two_phase_region_P = returnVector[9];

                bool twoPhaseRegionExists = (two_phase_region_P.size() > 1);

                if (twoPhaseRegionExists) {
                    // Two-phase region still exists → this is an AZEOTROPE, not a critical point

                    if (param.sw_azeotropicPoint == 1) {
                        // We're looking for the azeotrope: found it, stop
                        //azeotropeFound = true;
                        VLEDoesNotConvergeAnymore = true;
                        break;
                    }
                    else if (param.sw_criticalPoint == 1) {
                        // We're looking for the critical point: this is just an azeotrope, continue tracing
                        // The VLE curve continues beyond the azeotrope with inverted K values
                        continue;
                    }
                }
                else {
                    // No two-phase region: this is the CRITICAL POINT (or very close to it)

                    if (param.sw_criticalPoint == 1) {
                        // Found the critical point, stop
                        //criticalPointFound = true;
                        VLEDoesNotConvergeAnymore = true;
                        break;
                    }
                    else if (param.sw_azeotropicPoint == 1) {
                        // We wanted an azeotrope but hit the critical point: no azeotrope exists
                        VLEDoesNotConvergeAnymore = true;
                        break;
                    }
                }
            }

            // Calculate new stepsize
            actualsize = int(x.size());
            double stepsize_1 = 0.5 * std::max(abs(x[actualsize - 1](IndexOfSupercriticalComponent) - y[actualsize - 1](IndexOfSupercriticalComponent)), 0.0001);
            double stepsize_2 = 0.005 / (abs(x[actualsize - 1](IndexOfSupercriticalComponent) - y[actualsize - 1](IndexOfSupercriticalComponent)) + 0.00001);
            stepsize = std::min(stepsize_1, stepsize_2);
            if (stepsize < 0.0001 || abs(calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) - calculationScheme.originalConcentrations(0, IndexOfSupercriticalComponent)) < 0.01) {
                stepsize = 0.0001;
            }
            if (stepsize > 0.03) {
                stepsize = 0.03;
            }

            // Update x
            Eigen::VectorXd newxconcentration(2);
            newxconcentration(IndexOfSupercriticalComponent) = std::min(x[actualsize - 1](IndexOfSupercriticalComponent) + stepsize, 1.0);
            newxconcentration(IndexOfSubcriticalComponent) = std::max(x[actualsize - 1](IndexOfSubcriticalComponent) - stepsize, 0.0);
            calculationScheme.originalConcentrations.row(1) = newxconcentration;
            for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
            }
            calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();

            if (param.sw_Tbub == 1) {
                DeltaT = 0.0;
                if (PureTsat.size() > 0) {
                    DeltaT = 0.5 * (newxconcentration(0) - x[x.size() - 1](IndexOfSupercriticalComponent)) * (PureTsat[IndexOfSupercriticalComponent] - PureTsat[IndexOfSubcriticalComponent]);
                }
                int actualConcentrationIndex = calculationScheme.actualConcentrationIndices[i_calculation];
                int actualConcentrationIndex2 = calculationScheme.actualConcentrationIndices[i_calculation + 1];
                int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[i_calculation][0]];
                calculationScheme.temperatures[actualConcentrationIndex] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                calculationScheme.temperatures[actualConcentrationIndex2] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                calculationScheme.temperatures[actualReferenceStateConcentrationIndex] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                for (int g = 0; g < calculationScheme.TauConcentrationIndices.size(); g++) {
                    for (int h = 0; h < calculationScheme.TauConcentrationIndices[g].size(); h++) {
                        if (actualConcentrationIndex == calculationScheme.TauConcentrationIndices[g][h]) {
                            calculationScheme.TauTemperatures[g] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                        }
                    }
                }
            }
        }
    }
    std::vector<std::vector<Eigen::VectorXd>> returnVector;
    returnVector.push_back(x);
    returnVector.push_back(y);
    returnVector.push_back(K);
    returnVector.push_back(Pressures);
    returnVector.push_back(Temperatures);

    return returnVector;
}


// ====================== BUBBLE POINT FUNCTIONS ======================
/**
 * @brief Updates total pressure within the Bubble P algorithm.
 * * This function is to update the current total pressure P(k) within the Bubble P algorithm.
 * For this, it uses a linear interpolation strategy.
 */
inline void updateBubblePressure(calculation& calc, int i_calculation, int numberOfCalculations, double sum_yt, double& lowerP, double& lowerSum, double& upperP, double& upperSum) {
    if (sum_yt < 1.0 && sum_yt > lowerSum) {
        lowerP = calc.targetPressure(i_calculation);
        lowerSum = sum_yt;
    }
    else if (sum_yt > 1 && sum_yt < upperSum) {
        upperP = calc.targetPressure(i_calculation);
        upperSum = sum_yt;
    }

    double new_pressure = calc.targetPressure(i_calculation) * sum_yt;
    
    if (lowerP != 0.0 && upperP != 0.0) {
        new_pressure = lowerP + (1.0 - lowerSum) / (upperSum - lowerSum) * (upperP - lowerP);
    }
    calc.targetPressure(i_calculation) = new_pressure;
    calc.targetPressure(i_calculation + numberOfCalculations) = new_pressure;
}


/**
 * @brief Updates temperature within the Bubble T algorithm.
 * * This function is to update the current thermodynamic temperature T(k) within the Bubble T algorithm.
 * For this, it uses a linear interpolation strategy.
 */
inline void updateBubbleTemperature(parameters& param, calculation& calc, int i_calculation, int numberOfCalculations, double& sum_yt, double& last_sum, int& outlier_counter, double& lowerT, double& lowerSum, double& upperT, double& upperSum) {
    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];                          // actual concentration index of vapor phase
    int actualConcentrationIndex2 = calc.actualConcentrationIndices[i_calculation + numberOfCalculations];  // actual concentration index of liquid phase
    int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_calculation][0]];
    if (sum_yt < 1 && sum_yt > lowerSum) {
        lowerT = calc.temperatures[actualConcentrationIndex];
        lowerSum = sum_yt;
    }
    else if (sum_yt > 1 && sum_yt < upperSum) {
        upperT = calc.temperatures[actualConcentrationIndex];
        upperSum = sum_yt;
    }
    double new_temperature = calc.temperatures[actualConcentrationIndex];
    // Usually, new T is chosen based on a linear inerpolation strategy.
    // The routine in the else block is used alternatively.
    if (lowerT != 0.0 && upperT != 0.0) {
        new_temperature = lowerT + (1.0 - lowerSum) / (upperSum - lowerSum) * (upperT - lowerT);
    }
    else {
        if (sum_yt < 0.98) {
            sum_yt = 0.98;
        }
        else {
            int a = static_cast<int>(sum_yt);
            sum_yt = a + (sum_yt - a) / 10;
        }
        if (sum_yt > last_sum) {
            outlier_counter += 1;
            if (outlier_counter == 3) {
                calc.temperatures[actualConcentrationIndex] = 0.0;
                calc.targetTemperature(i_calculation) = 0.0;
                throw std::runtime_error("y_T has increased three times in a row. Bubble T algorithm has not converged.");
            }
        }
        else {
            outlier_counter = 0;
        }
        new_temperature = calc.temperatures[actualConcentrationIndex] * 1 / sum_yt;
    }
    last_sum = sum_yt;
    calc.temperatures[actualConcentrationIndex] = new_temperature;
    calc.temperatures[actualConcentrationIndex2] = new_temperature;
    calc.temperatures[actualReferenceStateConcentrationIndex] = new_temperature;
    for (int g = 0; g < calc.TauConcentrationIndices.size(); g++) {
        for (int h = 0; h < calc.TauConcentrationIndices[g].size(); h++) {
            if (actualConcentrationIndex == calc.TauConcentrationIndices[g][h]) {
                calc.TauTemperatures[g] = new_temperature;
            }
        }
    }
};


/**
 * @brief Alternative sampling-based Bubble P calculation strategy if initialization via Raoult's law fails.
 * * If initial K-values obtained using Raoult's law do not lead to convergence, this function performs a tracing strategy to find suitable K/y starting values ​​for calculating the bubble pressure for given x and T.
 * The following assumptions are made:
 * - binary mixture, only one component in supercritical state
 * - single connected VLE region
 * - unique critical endpoint
 * - monotonic x-P relationship
 */
inline void recoverBubblePressureBySampling(parameters& param, calculation& calc, int i_calculation, std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y, std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_P) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    double temperature = calc.temperatures[actualConcentrationIndex];                                   // temperature for which sampling is performed
    auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);   
    int index = it - calc.uniqueTemperatures.begin();

    // Supercritical / Subcritical detection
    int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
    int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
    auto classification = classifyComponents(subcalcindex_comp1, subcalcindex_comp2, subcalculations[subcalcindex_comp1].criticalT(0), subcalculations[subcalcindex_comp2].criticalT(0), temperature);
    int IndexOfSupercriticalComponent = classification.supercriticalIndex;
    int IndexOfSubcriticalComponent = classification.subcriticalIndex;
    int SubcalculationsIndexOfSupercriticalComponent = classification.supercriticalSubcalcIndex;
    int SubcalculationsIndexOfSubcriticalComponent = classification.subcriticalSubcalcIndex;
    
    double endx = calc.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

    if (sampling_x.size() == 0) {

        std::vector<std::vector<Eigen::VectorXd>> samplingResults;

        if (i_calculation == 0) {
            // i_calculation = 0 means this is the first calculation point and no other calculation results are availabe.
            // Therefore, the sampling procedure starts from the subcritical pure component state.
            double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
            startx(IndexOfSupercriticalComponent) = 0.0;
            startx(IndexOfSubcriticalComponent) = 1.0;
            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
            startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
            startK(IndexOfSubcriticalComponent) = 1.0;
            std::cout << "Start SamplingForVLE" << std::endl;
            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
        }
        else {
            // In this case the last succesfull calculation is used as starting point for the tracing procedure.
            std::cout << "Start SamplingForVLE" << std::endl;
            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, calc.originalConcentrations.row(i_calculation + numberOfCalculations - 1), calc.originalConcentrations.row(i_calculation - 1), calc.PartitionCoefficients.row(i_calculation - 1), calc.totalPressure(i_calculation - 1), {}, endx);
        }
        sampling_x = samplingResults[0];
        sampling_y = samplingResults[1];
        sampling_K = samplingResults[2];
        sampling_P = samplingResults[3];
    }
    double max_x = sampling_x[sampling_x.size() - 1](IndexOfSupercriticalComponent);
    if (calc.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent) <= max_x) {
        int index = 0;
        for (int i = 0; i < sampling_x.size(); i++) {
            if (sampling_x[i](IndexOfSupercriticalComponent) > calc.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent)) {
                if (i == 0) {
                    index = i;
                }
                else {
                    index = i - 1;
                }
                break;
            }
        }
        calc.originalConcentrations.row(i_calculation) = sampling_y[index];
        calc.PartitionCoefficients.row(i_calculation) = sampling_K[index];
        calc.targetPressure.row(i_calculation) = sampling_P[index];
        calculateBubblePoint(calc, i_calculation, 0.00001);
    }
    else {
        // Currently, the code throws an error when the user specified a x1 value and a temperature T for which no VLE occurs.
        // This might occur especially in the case of supercritical mixtures.
        // Alternatively, you could decide to set values to 0 here to indicate no solution if you don't want the code to stop and the other x1 values to be calculated as usual.

        std::stringstream ss;
        ss << "For given mixture, only one single phase is to be expected for all x_supercritical larger than "
            << max_x << ".";

        throw std::runtime_error(ss.str());;
    }
}


/**
 * @brief Alternative sampling-based Bubble T calculation strategy if initialization via Raoult's law fails.
 * * If initial K-values obtained using Raoult's law do not lead to convergence, this function performs a tracing strategy to find suitable K/y starting values ​for calculating the bubble temperature for given x and P.
 * The following assumptions are made:
 * - binary mixture, only one component in supercritical state
 * - single connected VLE region
 * - unique critical endpoint
 * - monotonic x-T relationship
 */
inline void recoverBubbleTemperatureBySampling(parameters& param, calculation& calc, int i_calculation, std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y, std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_T) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    double pressure = calc.targetPressure(i_calculation);
    std::cout << pressure << std::endl;
    double TsatComp1 = calc.SaturationTemperatures(i_calculation, 0);
    double TsatComp2 = calc.SaturationTemperatures(i_calculation, 1);
    std::vector<double> TsatPure;
    TsatPure.push_back(TsatComp1);
    TsatPure.push_back(TsatComp2);

    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    auto it = std::find(calc.uniquePressures.begin(), calc.uniquePressures.end(), pressure);
    int index = it - calc.uniquePressures.begin();
    
    // Supercritical / Subcritical detection
    int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
    int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
    auto classification = classifyComponents(subcalcindex_comp1, subcalcindex_comp2, subcalculations[subcalcindex_comp1].criticalP(0), subcalculations[subcalcindex_comp2].criticalP(0), pressure);
    int IndexOfSupercriticalComponent = classification.supercriticalIndex;
    int IndexOfSubcriticalComponent = classification.subcriticalIndex;
    int SubcalculationsIndexOfSupercriticalComponent = classification.supercriticalSubcalcIndex;
    int SubcalculationsIndexOfSubcriticalComponent = classification.subcriticalSubcalcIndex;

    if (sampling_x.size() == 0) {

        std::vector<std::vector<Eigen::VectorXd>> samplingResults;
        if (i_calculation == 0) {
            double endx = calc.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();
            double starttemperature = calc.SaturationTemperatures(i_calculation, IndexOfSubcriticalComponent);
            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
            startx(IndexOfSupercriticalComponent) = 0.0;
            startx(IndexOfSubcriticalComponent) = 1.0;
            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);

            startK(IndexOfSupercriticalComponent) = calc.SaturationPressures(i_calculation, IndexOfSupercriticalComponent) / calc.SaturationPressures(i_calculation, IndexOfSubcriticalComponent);
            startK(IndexOfSubcriticalComponent) = 1.0;
            std::cout << "Start sampling for Bubble T Calculation" << std::endl;
            samplingResults = samplingForVLE(calc.components, starttemperature, IndexOfSupercriticalComponent, startx, startx, startK, pressure, TsatPure, endx);
        }
        else {
            std::cout << "Start sampling for Bubble T Calculation" << std::endl;
            double endx = calc.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();
            samplingResults = samplingForVLE(calc.components, calc.targetTemperature(i_calculation - 1), IndexOfSupercriticalComponent, calc.originalConcentrations.row(i_calculation + numberOfCalculations - 1), calc.originalConcentrations.row(i_calculation - 1), calc.PartitionCoefficients.row(i_calculation - 1), calc.targetPressure(i_calculation), TsatPure, endx);
        }
        sampling_x = samplingResults[0];
        sampling_y = samplingResults[1];
        sampling_K = samplingResults[2];
        sampling_T = samplingResults[4];
    }
    double max_x = sampling_x[sampling_x.size() - 1](IndexOfSupercriticalComponent);
    if (calc.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent) <= max_x) {
        int index = 0;
        for (int i = 0; i < sampling_x.size(); i++) {
            if (sampling_x[i](IndexOfSupercriticalComponent) > calc.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent)) {
                if (i == 0) {
                    index = i;
                }
                else {
                    index = i - 1;
                }
                break;
            }
        }
        calc.originalConcentrations.row(i_calculation) = sampling_y[index];
        calc.PartitionCoefficients.row(i_calculation) = sampling_K[index];
        int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
        int actualConcentrationIndex2 = calc.actualConcentrationIndices[i_calculation + numberOfCalculations];
        int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_calculation][0]];
        calc.temperatures[actualConcentrationIndex] = sampling_T[index](0);
        calc.temperatures[actualConcentrationIndex2] = sampling_T[index](0);
        calc.temperatures[actualReferenceStateConcentrationIndex] = sampling_T[index](0);
        for (int g = 0; g < calc.TauConcentrationIndices.size(); g++) {
            for (int h = 0; h < calc.TauConcentrationIndices[g].size(); h++) {
                if (actualConcentrationIndex == calc.TauConcentrationIndices[g][h]) {
                    calc.TauTemperatures[g] = sampling_T[index](0);
                }
            }
        }
        calculateBubblePoint(calc, i_calculation, 0.00001);
    }
    else {
        throw std::runtime_error("For given mixture, only one single phase is to be expected.");
    }
}


/**
 * @brief Performs the two loops of the Bubble Point algorithm.
 * * This function performs the inner and outer loop of the Bubble Point Algorithm.
 * Initialization has to be done outside of the function, i.e. the function assumes 
 * that initial P, K and y guess are assumed to be already given. 
 * * @param calc: calculation to be calculated
 * @param i_calculation: index within _calculation
 * @param epsilon: tolerance
 * @param passInnerLoop: decide whether inner loop is skipped or not (cf. Elliot)
 */
inline void calculateBubblePoint(calculation& calc, int i_calculation, double epsilon, bool passInnerLoop) {
    
    param.sw_isotherm = 1;

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    bool bubblePorTFound = false;
    int counter = 0;
    int outlier_counter = 0;
    double last_sum = 100000;
    double TemperatureOrPressureWithSumyLowerNearestOne = 0.0;
    double sumyForTemperatureOrPressureWithSumyLowerNearestOne = 0.0;
    double TemperatureOrPressureWithSumyHigherNearestOne = 0.0;
    double sumyForTemperatureOrPressureWithSumyHigherNearestOne = 1000;
    std::vector<std::vector<double>> returnVector;
    std::vector<double> liquid_v;
    std::vector<double> liquid_P;

    ///// Outer loop /////
    constexpr int maxOuterIterations = 100;
    while (!bubblePorTFound && counter < maxOuterIterations) {
        calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
        calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

        // Liquid phase calculation (for Bubble P calculation, this calculation needs to be performed only once as x and T are specified)
        if (counter == 0 || param.sw_Tbub == 1) {
            returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
            liquid_v = returnVector[2];
            liquid_P = returnVector[3];
        }

        ///// Inner loop /////
        double sum_yt = calc.originalConcentrations.array().row(i_calculation).sum();   // sum of all vapor phase mole fractions
        double difference = 1000.0;                                                     // definition: abs(sum_yt_new - sum_yt_old)
        int inner_counter = 0;                                                          // number of iterations in inner loop
        double end_P = *std::min_element(liquid_P.begin(), liquid_P.end());
        
        while (difference > epsilon) {

            // Usually, calculation of the isotherm stops when P = 10 Pa is reached. In case of very low pressures, this endpoint has to be changed. 
            if (calc.targetPressure(i_calculation) < 10) {
                end_P = std::max(calc.targetPressure(i_calculation) - 1.0, 1.0);
            }

            // Vapor phase calculation
            calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
            calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;
            returnVector = calculateIsotherm(calc, i_calculation, end_P);
            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];
            std::vector<double> two_phase_region_P = returnVector[9];

            if (two_phase_region_P.size() <= 1 || vapor_P.size() == 0) {
                // In this case there is only one supercritical phase in equilibrium with the liquid phase
                vapor_v = returnVector[2];
                vapor_P = returnVector[3];
            }

            // Calculate Properties for current P, T, x
            calculateEqCondForGivenTPn(calc, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            if (calc.totalPressure(i_calculation) == 0.0 || passInnerLoop == true) {
                // In this case no equilibrium was found and we stop the calculation 
                break;
            }

            // Update K values and y for all components i
            for (int m = 0; m < calc.components.size() - 1; m++) {
                double fugacityCoefficientVapor = exp(calc.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, m));
                calc.PartitionCoefficients(i_calculation, m) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                calc.originalConcentrations(i_calculation, m) = calc.originalConcentrations(i_calculation + numberOfCalculations, m) * calc.PartitionCoefficients(i_calculation, m);
            }

            // Calculate difference = (sum_i y_i)^new - (sum_i y_i)^old, update sum and normalize y_i
            difference = abs(calc.originalConcentrations.array().row(i_calculation).sum() - sum_yt);
            sum_yt = calc.originalConcentrations.array().row(i_calculation).sum();
            calc.originalConcentrations.array().row(i_calculation) *= 1 / sum_yt;

            // Inner loop stops if either difference is smaller than epsilon or number of iterations is >= 10
            inner_counter += 1;
            if (inner_counter == 10) {
                break;
            }
        }

        passInnerLoop = false;

        if (calc.totalPressure(i_calculation) == 0.0) {
            if (param.sw_Tbub == 1) {
                calc.targetTemperature(i_calculation) = 0.0;
            }
            break;
        }

        // Check if sum_i y_i is close enough to target value 1
        if (abs(sum_yt - 1.0) < epsilon) {
            bubblePorTFound = true;
            if (param.sw_Tbub == 1) {
                int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
                calc.targetTemperature(i_calculation) = calc.temperatures[actualConcentrationIndex];
                calc.targetTemperature(i_calculation + numberOfCalculations) = calc.temperatures[actualConcentrationIndex];
            }
        }
        else {
            counter += 1;
            if (param.sw_Pbub == 1) {
                // Update pressure if Bubble P is to be calculated
                updateBubblePressure(calc, i_calculation, numberOfCalculations, sum_yt, TemperatureOrPressureWithSumyLowerNearestOne, sumyForTemperatureOrPressureWithSumyLowerNearestOne, TemperatureOrPressureWithSumyHigherNearestOne, sumyForTemperatureOrPressureWithSumyHigherNearestOne);
            }
            else if (param.sw_Tbub == 1) {
                // Update temperature if Bubble T is to be calculated
                updateBubbleTemperature(param, calc, i_calculation, numberOfCalculations, sum_yt, last_sum, outlier_counter, TemperatureOrPressureWithSumyLowerNearestOne, sumyForTemperatureOrPressureWithSumyLowerNearestOne, TemperatureOrPressureWithSumyHigherNearestOne, sumyForTemperatureOrPressureWithSumyHigherNearestOne);
            }
            sum_yt = calc.originalConcentrations.array().row(i_calculation).sum();
            calc.originalConcentrations.array().row(i_calculation) *= 1 / sum_yt;
        }
    }
    
    param.sw_isotherm = 0;
}


/**
 * @brief Calculates Bubble Pressures of a binary mixture at a fixed temperature for specified x1 values.
 * First, the algorithm is initialized using Raoult's law. 
 * If this fails, a tracing procedure is started. However, this has limited use cases; see the assumptions in the corresponding function description.
 * * @param calc: calculation to be calculated
 * @param tolerance: convergence criterion for sum_yt - 1.0
 */
inline void calculateBubblePoints(parameters& param, calculation& calc, double tolerance)
{
    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    std::vector<Eigen::VectorXd> sampling_x = {};
    std::vector<Eigen::VectorXd> sampling_y = {};
    std::vector<Eigen::VectorXd> sampling_K = {};
    std::vector<Eigen::VectorXd> sampling_P = {};
    std::vector<Eigen::VectorXd> sampling_T = {};

    std::cout << "Start Bubble Point Algorithm" << std::endl;

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

        ///// Initialization Routine 1: Shortcut K-ratio method / Raoult's law /////
        // y = K * x
        for (int m = 0; m < calc.components.size() - 1; m++) {
            calc.originalConcentrations(i_calculation, m) = calc.originalConcentrations(i_calculation + numberOfCalculations, m) * calc.PartitionCoefficients(i_calculation, m);
        }
        if (param.sw_Tbub == 1) {
            calc.originalConcentrations.array().row(i_calculation) *= 1 / calc.originalConcentrations.array().row(i_calculation).sum();
        }
        double startPressure = calc.targetPressure(i_calculation);
        calculateBubblePoint(calc, i_calculation, tolerance);

        ///// Initialization Routine 2 if failed /////
        if (param.sw_Pbub == 1 && calc.totalPressure(i_calculation) == 0.0) {

            // In this case no equilibrium was found using the initial values of routine 1
            if (int(calc.components.size() - 1) > 2) {
                throw std::runtime_error("Bubble Pressure Algorithm did not converge. Sampling currently only works for binary mixtures.");
            }

            recoverBubblePressureBySampling(param, calc, i_calculation, sampling_x, sampling_y, sampling_K, sampling_P);
        }
        else if (param.sw_Tbub == 1 && calc.targetTemperature(i_calculation) == 0.0) {
            // In this case no equilibrium was found using the initial values of routine 1

            if (int(calc.components.size() - 1) > 2) {
                throw std::runtime_error("Bubble Temperature Algorithm did not converge. Sampling currently only works for binary mixtures.");
            }

            recoverBubbleTemperatureBySampling(param, calc, i_calculation, sampling_x, sampling_y, sampling_K, sampling_T);
        }
    }
}



// ====================== DEW POINT FUNCTIONS ======================
/**
 * @brief Updates total pressure within the Dew P algorithm using linear interpolation.
 */
inline void updateDewPressure(calculation& calc, int i_calculation, int numberOfCalculations,
    double sum_xt, double& lowerP, double& lowerSum, double& upperP, double& upperSum)
{
    // Bracket update: track closest sums below and above 1
    if (sum_xt < 1.0 && sum_xt > lowerSum) {
        lowerP = calc.targetPressure(i_calculation);
        lowerSum = sum_xt;
    }
    else if (sum_xt > 1.0 && sum_xt < upperSum) {
        upperP = calc.targetPressure(i_calculation);
        upperSum = sum_xt;
    }

    // Default: successive substitution
    double new_pressure = calc.targetPressure(i_calculation) * (1.0 / sum_xt);

    // If bracket is available: use linear interpolation (Regula Falsi)
    if (lowerP != 0.0 && upperP != 0.0) {
        new_pressure = lowerP + (1.0 - lowerSum) / (upperSum - lowerSum) * (upperP - lowerP);
    }

    calc.targetPressure(i_calculation) = new_pressure;
    calc.targetPressure(i_calculation + numberOfCalculations) = new_pressure;
}


/**
 * @brief Updates temperature within the Dew T algorithm using linear interpolation.
 */
inline void updateDewTemperature(parameters& param, calculation& calc, int i_calculation, int numberOfCalculations,
    double& sum_xt, double& last_sum, int& outlier_counter,
    double& lowerT, double& lowerSum, double& upperT, double& upperSum)
{
    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    int actualConcentrationIndex2 = calc.actualConcentrationIndices[i_calculation + numberOfCalculations];
    int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_calculation][0]];

    // Bracket update
    // For Dew T: sum_xt < 1 means T too high, sum_xt > 1 means T too low
    // We still track "lowerSum" = closest sum < 1, "upperSum" = closest sum > 1
    if (sum_xt < 1.0 && sum_xt > lowerSum) {
        lowerT = calc.temperatures[actualConcentrationIndex];
        lowerSum = sum_xt;
    }
    else if (sum_xt > 1.0 && sum_xt < upperSum) {
        upperT = calc.temperatures[actualConcentrationIndex];
        upperSum = sum_xt;
    }

    double new_temperature = calc.temperatures[actualConcentrationIndex];

    // Linear interpolation if bracket is available
    if (lowerT != 0.0 && upperT != 0.0) {
        new_temperature = lowerT + (1.0 - lowerSum) / (upperSum - lowerSum) * (upperT - lowerT);
    }
    else {
        // Fallback: damped successive substitution
        if (sum_xt < 0.95) {
            sum_xt = 0.95;
        }
        else {
            int a = static_cast<int>(sum_xt);
            sum_xt = a + (sum_xt - a) / 10;
        }

        // Divergence detection
        if (sum_xt > last_sum) {
            outlier_counter += 1;
            if (outlier_counter == 3) {
                calc.temperatures[actualConcentrationIndex] = 0.0;
                calc.targetTemperature(i_calculation) = 0.0;
                throw std::runtime_error("x_sum has diverged three times in a row. Dew T algorithm has not converged.");
            }
        }
        else {
            outlier_counter = 0;
        }
        new_temperature = calc.temperatures[actualConcentrationIndex] * sum_xt;
    }

    last_sum = sum_xt;

    // Apply new temperature to all relevant indices
    calc.temperatures[actualConcentrationIndex] = new_temperature;
    calc.temperatures[actualConcentrationIndex2] = new_temperature;
    calc.temperatures[actualReferenceStateConcentrationIndex] = new_temperature;
    for (int g = 0; g < calc.TauConcentrationIndices.size(); g++) {
        for (int h = 0; h < calc.TauConcentrationIndices[g].size(); h++) {
            if (actualConcentrationIndex == calc.TauConcentrationIndices[g][h]) {
                calc.TauTemperatures[g] = new_temperature;
            }
        }
    }
}


/**
 * @brief Performs the two loops of the Dew Point algorithm.
 * * This function performs the inner and outer loop of the Dew Point Algorithm.
 * Initialization has to be done outside of the function, i.e. the function assumes
 * that initial P, K and y guess are assumed to be already given.
 * * @param calc: calculation to be calculated
 * @param i_calculation: index within _calculation
 * @param epsilon: tolerance
 * @param passInnerLoop: decide whether inner loop is skipped or not (cf. Elliot)
 */
inline void calculateDewPoint(calculation& calc, int i_calculation, double epsilon) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    bool dewPorTFound = false;
    int counter = 0;
    int outlier_counter = 0;
    double last_sum = 100000.0;

    // Bracket variables for Regula Falsi
    double bracketLowValue = 0.0;    // P or T where sum was closest below 1
    double bracketLowSum = 0.0;      // corresponding sum (< 1)
    double bracketHighValue = 0.0;   // P or T where sum was closest above 1
    double bracketHighSum = 1000.0;  // corresponding sum (> 1)

    std::vector<std::vector<double>> returnVector;
    std::vector<double> vapor_v;
    std::vector<double> vapor_P;

    ///// Outer loop /////
    constexpr int maxOuterIterations = 100;
    while (!dewPorTFound && counter < maxOuterIterations) {
        calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
        calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

        // Vapor phase calculation (for Dew P: only once, as y and T are fixed)
        if (counter == 0 || param.sw_Tdew == 1) {
            returnVector = calculateIsotherm(calc, i_calculation);
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
        }

        ///// Inner loop (iterating on x for fixed T, P, y) /////
        double sum_xt = calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
        double difference = 1000.0;
        int inner_counter = 0;

        while (difference > epsilon) {

            // Liquid phase calculation
            calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
            calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;
            returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            // Calculate properties for current P, T, x
            calculateEqCondForGivenTPn(calc, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            if (calc.totalPressure(i_calculation) == 0.0) {
                break;
            }

            // Update K values and x
            for (int m = 0; m < calc.components.size() - 1; m++) {
                double fugacityCoefficientVapor = exp(calc.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, m));
                calc.PartitionCoefficients(i_calculation, m) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = calc.originalConcentrations(i_calculation, m) / calc.PartitionCoefficients(i_calculation, m);
            }

            difference = abs(calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum() - sum_xt);
            sum_xt = calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            calc.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1.0 / sum_xt;

            inner_counter += 1;
            if (inner_counter == 10) {
                break;
            }
        }

        if (calc.totalPressure(i_calculation) == 0.0) {
            if (param.sw_Tdew == 1) {  // FIXED: war sw_Tbub
                calc.targetTemperature(i_calculation) = 0.0;
            }
            break;
        }

        // Check convergence: sum_i x_i close to 1?
        if (abs(sum_xt - 1.0) < epsilon) {
            dewPorTFound = true;
            if (param.sw_Tdew == 1) {
                int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
                calc.targetTemperature(i_calculation) = calc.temperatures[actualConcentrationIndex];
                calc.targetTemperature(i_calculation + numberOfCalculations) = calc.temperatures[actualConcentrationIndex];
            }
        }
        else {
            counter += 1;
            if (param.sw_Pdew == 1) {
                updateDewPressure(calc, i_calculation, numberOfCalculations, sum_xt,
                    bracketLowValue, bracketLowSum, bracketHighValue, bracketHighSum);
            }
            else if (param.sw_Tdew == 1) {
                updateDewTemperature(param, calc, i_calculation, numberOfCalculations, sum_xt,
                    last_sum, outlier_counter, bracketLowValue, bracketLowSum, bracketHighValue, bracketHighSum);
            }
            sum_xt = calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            calc.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1.0 / sum_xt;
        }
    }

    if (!dewPorTFound) {
        throw std::runtime_error("Dew point algorithm did not converge.");
    }
}


/**
 * @brief Calculates Dew Pressures of a binary mixture at a fixed temperature for specified x1 values.
 * * @param calc: calculation to be calculated
 * @param tolerance: convergence criterion for sum_yt - 1.0
 */
inline void calculateDewPoints(parameters& param, calculation& calc, double tolerance) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

        ///// Initialization Routine 1: Raoult's law /////
        for (int m = 0; m < calc.components.size() - 1; m++) {
            calc.originalConcentrations(i_calculation + numberOfCalculations, m) = calc.originalConcentrations(i_calculation, m) / calc.PartitionCoefficients(i_calculation, m);
        }
        // Perform normalization when calculating Dew T because of possible inaccuracies
        if (param.sw_Tdew == 1) {
            calc.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
        }
        calculateDewPoint(calc, i_calculation, tolerance);

        ///// Initialization Routine 2: currently not implemented /////
    }
}



// ====================== PT FLASH FUNCTIONS ======================
/**
 * @brief Initializes partition coefficients and phase compositions for LLE flash calculation.
 *
 * This function determines initial K-values and phase compositions for LLE by:
 * 1. Computing activity coefficients (gamma) from infinite-dilution fugacity coefficients
 * 2. Estimating initial phase split based on gamma * z
 * 3. Iteratively finding a valid two-phase equilibrium condition
 * 4. Computing an initial theta guess from the Rachford-Rice equation
 *
 * @param param: global parameters
 * @param calc: calculation object
 * @param i_calculation: index of the current calculation point
 * @param[out] thetaGuess: initial vapor/second-phase fraction estimate
 * @param[out] skipThetaCalculation: set to true if thetaGuess was determined here
 */
inline double initializeLLEFlash(parameters& param, calculation& calc, int i_calculation, double& thetaGuess, bool& skipThetaCalculation)
{
    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = int(calc.components.size()) - 1;
    double pressure = calc.targetPressure(i_calculation);

    // Step 1: Compute activity coefficients from feed composition (L1 phase)
    std::vector<double> gammas;

    calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
    calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

    std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
    std::vector<double> liquid1_v = returnVector[2];
    std::vector<double> liquid1_P = returnVector[3];

    double v = calculatevForGivenTPn(calc, i_calculation + numberOfCalculations, liquid1_v, liquid1_P, pressure);
    calc.molarVolume(i_calculation + numberOfCalculations) = v;
    calculateEOS(param, calc, i_calculation + numberOfCalculations);

    for (int m = 0; m < numberOfOriginalComponents; m++) {
        // Set L2 phase to pure component m
        calc.originalConcentrations.row(i_calculation).setZero();
        calc.originalConcentrations(i_calculation, m) = 1.0;

        calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
        calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

        std::vector<std::vector<double>> retVec = calculateIsotherm(calc, i_calculation);
        std::vector<double> liquid2_v = retVec[2];
        std::vector<double> liquid2_P = retVec[3];

        double v_pure = calculatevForGivenTPn(calc, i_calculation, liquid2_v, liquid2_P, pressure);
        calc.molarVolume(i_calculation) = v_pure;
        calculateEOS(param, calc, i_calculation);

        double gammamL1 = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, m) - calc.lnPhiTotal(i_calculation, m));
        gammas.push_back(gammamL1);
    }

    // Step 2: Estimate initial L2 composition from gamma * z
    for (int m = 0; m < numberOfOriginalComponents; m++) {
        calc.originalConcentrations(i_calculation, m) = gammas[m] * calc.originalConcentrations(i_calculation + numberOfCalculations, m);
    }
    double sum = calc.originalConcentrations.array().row(i_calculation).sum();
    calc.originalConcentrations.array().row(i_calculation) *= 1.0 / sum;

    // Compute fugacity coefficients for estimated L2 phase
    calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
    calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

    returnVector = calculateIsotherm(calc, i_calculation);
    std::vector<double> liquid2_v = returnVector[2];
    std::vector<double> liquid2_P = returnVector[3];

    v = calculatevForGivenTPn(calc, i_calculation, liquid2_v, liquid2_P, pressure);
    calc.molarVolume(i_calculation) = v;
    calculateEOS(param, calc, i_calculation);

    // Step 3: Compute initial K-values and identify dominant components
    calc.PartitionCoefficients = Eigen::MatrixXd(int(calc.originalConcentrations.rows()), numberOfOriginalComponents);

    int indexWithMaxKi = -1;
    double maxPartitionCoefficient = 0.0;
    int indexWithMinKi = -1;
    double minPartitionCoefficient = 100000.0;

    for (int m = 0; m < numberOfOriginalComponents; m++) {
        double fugacityCoefficientLiquid1 = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, m));
        double fugacityCoefficientLiquid2 = exp(calc.lnPhiTotal(i_calculation, m));
        double K = fugacityCoefficientLiquid1 / fugacityCoefficientLiquid2;

        if (K > maxPartitionCoefficient) {
            indexWithMaxKi = m;
            maxPartitionCoefficient = K;
        }
        if (K < minPartitionCoefficient) {
            indexWithMinKi = m;
            minPartitionCoefficient = K;
        }
        calc.PartitionCoefficients(i_calculation, m) = K;
    }

    // Step 4: Iteratively find valid two-phase equilibrium starting point
    constexpr int maxInitIterations = 100;
    bool eqCondFound = false;

    for (int counter = 0; counter < maxInitIterations; counter++) {
        calc.originalConcentrations.row(i_calculation).setZero();
        calc.originalConcentrations.row(i_calculation + numberOfCalculations).setZero();

        calc.originalConcentrations(i_calculation, indexWithMaxKi) = 0.98 - counter * 0.002;
        calc.originalConcentrations(i_calculation, indexWithMinKi) = 0.02 + counter * 0.002;
        calc.originalConcentrations(i_calculation + numberOfCalculations, indexWithMaxKi) = 0.02 + counter * 0.002;
        calc.originalConcentrations(i_calculation + numberOfCalculations, indexWithMinKi) = 0.98 - counter * 0.002;

        calc.PartitionCoefficients.row(i_calculation).setZero();

        calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
        calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

        returnVector = calculateIsotherm(calc, i_calculation);
        std::vector<double> phase2_v = returnVector[2];
        std::vector<double> phase2_P = returnVector[3];

        returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
        std::vector<double> phase1_v = returnVector[2];
        std::vector<double> phase1_P = returnVector[3];

        calculateEqCondForGivenTPn(calc, i_calculation, phase2_v, phase2_P, phase1_v, phase1_P, false);

        if (calc.totalPressure(i_calculation) != 0.0) {
            eqCondFound = true;
            break;
        }
    }

    if (!eqCondFound) {
        throw std::runtime_error("Initialization scheme for LLE failed, no LLE calculation possible.");
    }

    // Step 5: Compute final K-values for dominant components and theta guess
    double fugCoeffVapor = exp(calc.lnPhiTotal(i_calculation, indexWithMaxKi));
    double fugCoeffLiquid = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMaxKi));
    calc.PartitionCoefficients(i_calculation, indexWithMaxKi) = fugCoeffLiquid / fugCoeffVapor;

    fugCoeffVapor = exp(calc.lnPhiTotal(i_calculation, indexWithMinKi));
    fugCoeffLiquid = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMinKi));
    calc.PartitionCoefficients(i_calculation, indexWithMinKi) = fugCoeffLiquid / fugCoeffVapor;

    thetaGuess = (calc.overallConcentrations(i_calculation, indexWithMaxKi) * (1.0 - 1.0 / calc.PartitionCoefficients(i_calculation, indexWithMaxKi))
        + calc.overallConcentrations(i_calculation, indexWithMinKi) * calc.PartitionCoefficients(i_calculation, indexWithMinKi))
        / (calc.overallConcentrations(i_calculation, indexWithMinKi) + calc.overallConcentrations(i_calculation, indexWithMaxKi));

    return thetaGuess;
}


/**
 * @brief Initialization Routine 2 within function runPTFlash that is used when initialization with Raoult's K fails
 */
inline void initializeAndSolveVLEFlashBySampling(parameters& param, calculation& calc, int i_calculation,
    std::vector<Eigen::VectorXd>& sampling_x, std::vector<Eigen::VectorXd>& sampling_y,
    std::vector<Eigen::VectorXd>& sampling_K, std::vector<Eigen::VectorXd>& sampling_P)
{
    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    double temperature = calc.temperatures[actualConcentrationIndex];
    auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);
    int index = it - calc.uniqueTemperatures.begin();

    // 1. Supercritical/Subcritical detection
    int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
    int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
    auto classification = classifyComponents(subcalcindex_comp1, subcalcindex_comp2, subcalculations[subcalcindex_comp1].criticalT(0), subcalculations[subcalcindex_comp2].criticalT(0), temperature);
    int IndexOfSupercriticalComponent = classification.supercriticalIndex;
    int IndexOfSubcriticalComponent = classification.subcriticalIndex;
    int SubcalculationsIndexOfSupercriticalComponent = classification.supercriticalSubcalcIndex;
    int SubcalculationsIndexOfSubcriticalComponent = classification.subcriticalSubcalcIndex;


    // 2. Perform sampling if not yet done
    double endx = calc.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

    if (sampling_x.size() == 0) {

        std::vector<std::vector<Eigen::VectorXd>> samplingResults;
        param.sw_Pbub = 1;
        if (i_calculation == 0) {
            double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
            startx(IndexOfSupercriticalComponent) = 0.0;
            startx(IndexOfSubcriticalComponent) = 1.0;
            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
            startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
            startK(IndexOfSubcriticalComponent) = 1.0;
            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
        }
        else {
            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, calc.originalConcentrations.row(i_calculation + numberOfCalculations - 1), calc.originalConcentrations.row(i_calculation - 1), calc.PartitionCoefficients.row(i_calculation - 1), calc.totalPressure(i_calculation - 1), {}, endx);
        }
        sampling_x = samplingResults[0];
        sampling_y = samplingResults[1];
        sampling_K = samplingResults[2];
        sampling_P = samplingResults[3];
        param.sw_Pbub = 0;
    }
    
    auto range = determineSamplingRange(sampling_x, sampling_y, sampling_P, IndexOfSupercriticalComponent);

    double z = calc.overallConcentrations(i_calculation, IndexOfSupercriticalComponent);
    double P_target = calc.targetPressure(i_calculation);

    auto searchResult = findTwoPhaseSegment(sampling_x, sampling_y, sampling_P, IndexOfSupercriticalComponent, z, P_target, range);

    bool AggregationStateDetermined = false;
    if (searchResult.twoPhaseFound) {
        // Set initial values from sampling and calculate Flash
        calc.originalConcentrations.row(i_calculation) = sampling_y[searchResult.twoPhaseIndex];
        calc.originalConcentrations.row(i_calculation + numberOfCalculations) = sampling_x[searchResult.twoPhaseIndex];
        calc.PartitionCoefficients.row(i_calculation) = sampling_K[searchResult.twoPhaseIndex];
        std::cout << "Now compute Flash after Sampling" << std::endl;
        calculatePTFlash(calc, i_calculation, 0.001, false, 0.0, true);
        AggregationStateDetermined = true;
    }
    else if (searchResult.zInAnySegment) {
        if (P_target < searchResult.P_low) {
            // single gaseous phase
            for (int m = 0; m < calc.components.size() - 1; ++m)
            {
                calc.originalConcentrations(i_calculation, m) = calc.overallConcentrations(i_calculation, m);
                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = 0.0;
            }
            AggregationStateDetermined = true;
        }
        else if (P_target > searchResult.P_high) {
            // single liquid phase
            for (int m = 0; m < calc.components.size() - 1; ++m)
            {
                calc.originalConcentrations(i_calculation, m) = 0.0;
                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = calc.overallConcentrations(i_calculation, m);
            }
            AggregationStateDetermined = true;
        }
    }

    if (AggregationStateDetermined == false) {
        throw std::runtime_error("An error occured, aggregation state of mixture could not be determined.");
    }
}


/**
 * @brief Determines phase state (VLE two-phase, single vapor, single liquid) and phase compositions
 *        for a given (z, T, P) by solving the RR equation (or more precisely its associated polynom).
 * The K values are initialized via a Bubble and Dew Point calculation. 
 * If UseGivenK = True, the initialization procedure is skipped and K values stored in calc.PartitionCoefficients are used as starting values for RR.
 */
inline void calculatePTFlash(calculation calc, int i_calculation, double ConvergenceCriteria, bool skipThetaCalculation, double thetaGuess, bool UseGivenK) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    // Associated polynom of RR function
    auto f_RRap = [&calc, i_calculation](double theta) {
        double functionValue = 0.0;
        for (int m = 0; m < calc.components.size() - 1; m++) {
            double val1 = double(calc.overallConcentrations(i_calculation, m)) * (calc.PartitionCoefficients(i_calculation, m) - 1.0) / (1.0 + theta * (calc.PartitionCoefficients(i_calculation, m) - 1.0));
            double val2 = 1.0;
            for (int n = 0; n < calc.components.size() - 1; n++) {
                val2 *= (1.0 + theta * (calc.PartitionCoefficients(i_calculation, n) - 1.0));
            }
            functionValue += val1 * val2;
        }
        return functionValue;
        };

    // RR function
    auto f_RR = [&calc, i_calculation](double theta) {
        double functionValue = 0.0;
        for (int m = 0; m < calc.components.size() - 1; m++) {
            functionValue += ((calc.PartitionCoefficients(i_calculation, m) - 1.0) * calc.overallConcentrations(i_calculation, m)) / (1.0 + (calc.PartitionCoefficients(i_calculation, m) - 1.0) * theta);
        }
        return functionValue;
        };

    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_calculation][0]];
    double temperature = calc.temperatures[actualConcentrationIndex];
    double pressure = calc.targetPressure(i_calculation);

    // Stability analysis
    bool SuccessfullyFoundInitialK = false;

    if (UseGivenK == true) {
        SuccessfullyFoundInitialK = true;
    }

    try {
        if (calc.calculationType == "VL" && UseGivenK == false) {
            std::vector<Eigen::VectorXd> sampling_x = {};
            std::vector<Eigen::VectorXd> sampling_y = {};
            std::vector<Eigen::VectorXd> sampling_K = {};
            std::vector<Eigen::VectorXd> sampling_P = {};
            std::vector<Eigen::VectorXd> sampling_T = {};

            // Set x_i = z_i and perform Bubble P calculation
            param.sw_Pbub = 1;

            std::vector<double> pressureVector;
            pressureVector.push_back(double(pressure));
            std::vector<std::vector<double>> concentrationx1(1);
            for (int m = 0; m < calc.components.size() - 1; m++) {
                concentrationx1[0].push_back(calc.overallConcentrations(i_calculation, m));
            }

            calculation calculationScheme = loadSubcalculation(calc.components, { double(temperature) }, concentrationx1, pressureVector);
            int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
            calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});

            Eigen::VectorXd TargetPressureAccordingToRaoult = (calc.overallConcentrations * calc.SaturationPressures.transpose()).diagonal();

            calculationScheme.PartitionCoefficients = Eigen::MatrixXd::Zero(2, 2);
            for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                calculationScheme.PartitionCoefficients(0, i) = calc.SaturationPressures(i_calculation, i) / TargetPressureAccordingToRaoult(i_calculation);
                calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
            }
            calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
            int numberOfOriginalComponents = calculationScheme.components.size() - 1;
            calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
            calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
            calculationScheme.calculationType = "VL";
            calculationScheme.originalNumberOfCalculations = 2;

            for (int i = 0; i < numberOfOriginalComponents; i++) {
                int index = calculationScheme.components[i]->index;
                calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
            }

            calculateBubblePoint(calculationScheme, 0, 0.0001);

            double Pbub = calculationScheme.totalPressure(1);
            Eigen::VectorXd Kbub = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();

            param.sw_Pbub = 0;

            if (Pbub != 0.0) {

                // Set y_i = z_i and perform Dew P calculation
                param.sw_Pdew = 1;
                calculationScheme.originalConcentrations.row(0) = calc.overallConcentrations.row(i_calculation);
                double dewPressureGuess = 1 / ((calculationScheme.originalConcentrations.row(0).array() / calc.SaturationPressures.row(i_calculation).array()).matrix().sum());
                calculationScheme.targetPressure(0) = dewPressureGuess;
                calculationScheme.targetPressure(1) = dewPressureGuess;
                calculationScheme.totalPressure(0) = dewPressureGuess;
                calculationScheme.totalPressure(1) = dewPressureGuess;
                calculationScheme.PartitionCoefficients.row(0) = calc.SaturationPressures.row(i_calculation) / dewPressureGuess;
                for (int m = 0; m < calculationScheme.components.size() - 1; m++) {
                    calculationScheme.originalConcentrations(1, m) = calculationScheme.originalConcentrations(0, m) / calculationScheme.PartitionCoefficients(0, m);
                }
                if (param.sw_Tdew == 1) {
                    calculationScheme.originalConcentrations.array().row(1) *= 1 / calculationScheme.originalConcentrations.array().row(1).sum();
                }
                calculateDewPoint(calculationScheme, 0, 0.0001);

                double Pdew = calculationScheme.totalPressure(1);
                Eigen::VectorXd Kdew = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();
                param.sw_Pdew = 0;

                // Get and assign interpolated K
                if (Pbub != 0.0 && Pdew != 0.0) {
                    SuccessfullyFoundInitialK = true;
                    Eigen::VectorXd K = interpolateK(pressure, Pdew, Pbub, Kdew, Kbub);
                    calc.PartitionCoefficients.row(i_calculation) = K;
                }
                if ((Pbub == 0.0 || Pdew == 0.0) && param.sw_excess == 1) {
                    for (int m = 0; m < calc.components.size() - 1; m++) {
                        calc.originalConcentrations(i_calculation, m) = 0.0;
                        calc.originalConcentrations(i_calculation + numberOfCalculations, m) = 0.0;
                    }
                }
            }
        }
    }

    catch (...) {
        SuccessfullyFoundInitialK = false;
    }

    if (SuccessfullyFoundInitialK == true) {
        int numberOfConvergedCriteria = 0;
        int numberOfIterations = 0;
        int maxNumberOfIterations = 75;
        float previous_theta = 0;
        std::vector<double> previousDelta(int(calc.components.size()), 0.0);


        while (numberOfConvergedCriteria != int(calc.components.size())) {
            numberOfConvergedCriteria = 0;
            numberOfIterations += 1;

            // Theta from RRap
            if (skipThetaCalculation == false) {
                if (param.sw_excess == 1) {
                    thetaGuess = ITP(f_RRap, 0.0, 1.0, std::nan("1"), std::nan("1"), 1e-10, 0.000001, std::nan("1"), 2, 1, false);
                    if (std::isnan(thetaGuess)) {
                        if (f_RR(0) < 0) {
                            // single liquid phase
                            for (int m = 0; m < calc.components.size() - 1; m++) {
                                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = calc.overallConcentrations(i_calculation, m);
                                calc.originalConcentrations(i_calculation, m) = 0.0;
                            }
                        }
                        else if (f_RR(1) > 0) {
                            // single gas phase
                            for (int m = 0; m < calc.components.size() - 1; m++) {
                                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = 0.0;
                                calc.originalConcentrations(i_calculation, m) = calc.overallConcentrations(i_calculation, m);
                            }
                        }
                        break;
                    }
                }
                else {
                    thetaGuess = ITP(f_RRap, 0.0, 1.0, std::nan("1"), std::nan("1"), 1e-10, 0.000001);
                }
                //double RR_value = f_RR(thetaGuess);
            }
            if (numberOfIterations == maxNumberOfIterations) {
                break;
            }

            // x_i and y_i from mass balance and possible normalization
            for (int m = 0; m < calc.components.size() - 1; m++) {
                calc.originalConcentrations(i_calculation + numberOfCalculations, m) = calc.overallConcentrations(i_calculation, m) / (1.0 + thetaGuess * (calc.PartitionCoefficients(i_calculation, m) - 1.0));
                calc.originalConcentrations(i_calculation, m) = calc.originalConcentrations(i_calculation + numberOfCalculations, m) * calc.PartitionCoefficients(i_calculation, m);
            }
            double sum = calc.originalConcentrations.array().row(i_calculation).sum();
            calc.originalConcentrations.array().row(i_calculation) *= 1 / sum;
            sum = calc.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            calc.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / sum;

            // Calculate Isotherms, v and fugacity coefficients with EoS
            calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
            calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

            std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation);
            std::vector<double> vapor_v; // For LL this is actually L2
            std::vector<double> vapor_P;
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
            if (calc.calculationType == "VL") {
                if (vapor_v.size() == 0) {
                    // In this case there is only one supercritical phase in equilibrium with the liquid phase
                    vapor_v = returnVector[2];
                    vapor_P = returnVector[3];
                }
            }
            else if (calc.calculationType == "LL") {
                vapor_v = returnVector[2];
                vapor_P = returnVector[3];
            }
            returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            calculateEqCondForGivenTPn(calc, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            if (calc.totalPressure(i_calculation) == 0.0) {
                calc.originalConcentrations.row(i_calculation) = calc.overallConcentrations.row(i_calculation);
                calc.originalConcentrations.row(i_calculation + numberOfCalculations) = calc.overallConcentrations.row(i_calculation);
                break;
            }

            for (int m = 0; m < calc.components.size() - 1; m++) {
                double fugacityCoefficientVapor = exp(calc.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(calc.lnPhiTotal(i_calculation + numberOfCalculations, m));
                double newPartitionCoefficient = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                double delta_new = newPartitionCoefficient - calc.PartitionCoefficients(i_calculation, m);
                double delta_old = previousDelta[m];
                bool signChange = (delta_new * delta_old < 0.0);
                if (abs(delta_new / newPartitionCoefficient) < ConvergenceCriteria) {
                    numberOfConvergedCriteria += 1;
                }
                if (signChange) {
                    newPartitionCoefficient = 0.5 * newPartitionCoefficient + 0.5 * calc.PartitionCoefficients(i_calculation, m);
                }
                calc.PartitionCoefficients(i_calculation, m) = newPartitionCoefficient;
                previousDelta[m] = delta_new;
            }
            if (abs((calc.targetPressure(i_calculation) - pressure) / calc.targetPressure(i_calculation)) < ConvergenceCriteria) {
                numberOfConvergedCriteria += 1;
            }
            skipThetaCalculation = false;
            previous_theta = thetaGuess;
        }
        // For some systems small jumps between the total Pressure / K's have been observed and the convergence criterion above is not fullfilled.
        // The algorithm then gets stuck in an infinite loop or converges only very slowly.
        // For this reason, the loop stops after 50 iterations and then another convergence criterion is checked and only if this one is also not met, _calculation.totalPressure(i_calculation) = 0.0 which means no convergence.
        if (numberOfIterations >= maxNumberOfIterations) {
            if (!(abs(1 - calc.originalConcentrations.row(i_calculation + numberOfCalculations).dot(calc.PartitionCoefficients.row(i_calculation))) <= 0.001 || abs(thetaGuess - previous_theta) <= 0.01)) {
                calc.totalPressure(i_calculation) = 0.0;
            }
        }
    }
    else {
        calc.totalPressure(i_calculation) = 0.0;
    }
}


/**
 * @brief Determines phase state (VLE two-phase, single vapor, single liquid) and phase compositions
 *        for multiple (z, T, P) triplets by solving the RR equation (or more precisely its associated polynom).
 */
inline void runPTFlash(parameters& param, calculation& calc) {

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = int(calc.components.size()) - 1;

    std::vector<Eigen::VectorXd> sampling_x = {};
    std::vector<Eigen::VectorXd> sampling_y = {};
    std::vector<Eigen::VectorXd> sampling_K = {};
    std::vector<Eigen::VectorXd> sampling_P = {};
    std::vector<Eigen::VectorXd> sampling_T = {};

    std::cout << "Start pT Flash Algorithm" << std::endl;

    calc.PartitionCoefficients = Eigen::MatrixXd(int(calc.originalConcentrations.rows()), numberOfOriginalComponents);

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

        bool skipThetaCalculation = false;
        bool useGivenK = false;
        double thetaGuess = 0.0;

        if (param.useGivenInitialK == true && !(calc.calculatedPartitionCoefficients.row(i_calculation).isZero(1e-7))) {
            calc.PartitionCoefficients.row(i_calculation) = calc.calculatedPartitionCoefficients.row(i_calculation);
            useGivenK = true;
        }

        int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
        double temperature = calc.temperatures[actualConcentrationIndex];
        double pressure = calc.targetPressure(i_calculation);

        if (calc.calculationType == "LL") {

            if (param.useGivenInitialK == false) {
                // Initialization scheme for LLE
                thetaGuess = initializeLLEFlash(param, calc, i_calculation, thetaGuess, skipThetaCalculation);
                skipThetaCalculation = true;
            }
            calculatePTFlash(calc, i_calculation, 0.001, skipThetaCalculation, thetaGuess, true);
        }

        else if (calc.calculationType == "VL") {
            // Attempt 1: Direct flash with Raoult initialization
            bool flashSucceeded = false;
            if (sampling_x.empty()) {
                calculatePTFlash(calc, i_calculation, 0.001, skipThetaCalculation, thetaGuess, useGivenK);
                flashSucceeded = (calc.totalPressure(i_calculation) != 0.0);
            }

            // Attempt 2: Sampling-based initialization
            auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);
            int index = it - calc.uniqueTemperatures.begin();
            int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
            int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
            if (!flashSucceeded && !param.noTracing) {

                if (int(calc.components.size() - 1) > 2) {
                    throw std::runtime_error("PT Flash Algorithm did not converge. Sampling currently only works for binary mixtures.");
                }

                initializeAndSolveVLEFlashBySampling(param, calc, i_calculation, sampling_x, sampling_y, sampling_K, sampling_P);
            }
        }
    }
}



// ====================== CRITICAL / AZEOTROPIC POINT FUNCTIONS ======================
/**
 * @brief Calculates the critical point for a calculation point within structure calculation.
 * The specified variable is x and the algorithm finds corresponding T and P where only one supercritical phase is observed.
 */
inline void findCriticalPointForSpecifiedx(calculation& calc, int i_calculation) {

    std::vector<double> TemperatureGuessesTried = std::vector<double>();
    int numberOfStatePoints = int(calc.originalNumberOfCalculations / 2);
    
    int i_calculation_vapor = i_calculation;
    int i_calculation_liquid = i_calculation + numberOfStatePoints;
    int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
    int actualConcentrationIndex2 = calc.actualConcentrationIndices[i_calculation + numberOfStatePoints];
    int actualReferenceStateConcentrationIndex = calc.actualConcentrationIndices[calc.referenceStateCalculationIndices[i_calculation][0]];
    double TemperatureToSave = calc.temperatures[actualConcentrationIndex]; // temperature originally given should later be restored
    double TemperatureGuess = 100; // inital value at the moment: T = 100 K
    double TemperatureLimit1 = 100;
    double TemperatureLimit2 = 1e8;
    bool TemperatureLimit2Found = false;

    std::vector<std::vector<double>> returnVector;
    std::vector<double> v;
    std::vector<double> P;
    std::vector<double> liquid_v;
    std::vector<double> liquid_P;
    std::vector<double> vapor_v;
    std::vector<double> vapor_P;
    std::vector<double> two_phase_region_v;
    std::vector<double> two_phase_region_P;
    std::vector<double> P_repulsive;
    std::vector<double> P_attractive;

    double stepsize = 50;
    bool StartingTemperatureFound = false;
    double criticalPGuess = 0.0;

    while (abs(TemperatureLimit2 - TemperatureLimit1) > 0.0001) {
        calc.temperatures[actualConcentrationIndex] = TemperatureGuess;
        calc.temperatures[actualConcentrationIndex2] = TemperatureGuess;
        calc.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
        calc.TauTemperatures[0] = TemperatureGuess;
        int size_vapor = 0;
        int size_two_phase_region = 0;
        try {
            returnVector = calculateIsotherm(calc, i_calculation, 10.0);
            v = returnVector[0];
            P = returnVector[1];
            liquid_v = returnVector[2];
            liquid_P = returnVector[3];
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
            P_repulsive = returnVector[6];
            P_attractive = returnVector[7];
            size_vapor = int(vapor_v.size());
            if (size_vapor == 0 || size_vapor == 1) {
                if (StartingTemperatureFound == false) {
                    TemperatureGuess += 50;
                }
                else {
                    TemperatureLimit2 = TemperatureGuess;
                    TemperatureLimit2Found = true;
                    stepsize = abs(TemperatureLimit2 - TemperatureLimit1) / 10;
                    TemperatureGuess = TemperatureLimit1 + stepsize;
                }
            }
            else {
                StartingTemperatureFound = true;
                criticalPGuess = (liquid_P[liquid_P.size() - 1] + vapor_P[0]) / 2;
                TemperatureLimit1 = TemperatureGuess;
                TemperatureGuess = TemperatureGuess + stepsize;
            }
        }
        catch (...) {
            TemperatureGuess += 50;
        }
    }
    calc.criticalT(i_calculation_vapor) = TemperatureLimit2;
    calc.criticalT(i_calculation_liquid) = TemperatureLimit2;
    calc.criticalP(i_calculation_vapor) = criticalPGuess;
    calc.totalPressure(i_calculation_liquid) = criticalPGuess;
    calc.criticalP(i_calculation_liquid) = criticalPGuess;

    calc.temperatures[actualConcentrationIndex] = TemperatureToSave;
    calc.temperatures[actualConcentrationIndex2] = TemperatureToSave;
    calc.temperatures[actualReferenceStateConcentrationIndex] = TemperatureToSave;
    calc.TauTemperatures[0] = TemperatureToSave;
}

/**
 * @brief Implementation of a binary mixture's critical point calculation for several specified x1 values.
 */
inline void calculateCriticalPointForSpecifiedx(parameters& param, calculation& calc) {

    calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
    calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
        findCriticalPointForSpecifiedx(calc, i_calculation);
    }
}


/**
 * @brief Calculates the critical or azeotropic point of a binary mixture at given temperature
 *        using a VLE curve tracing strategy.
 *
 * Strategy:
 * The function traces the VLE (vapor-liquid equilibrium) curve starting from the pure
 * subcritical component (x_supercritical = 0) towards increasing mole fractions of the
 * supercritical component. Along this curve, the compositions of coexisting phases (x, y)
 * are monitored. As the curve approaches either an azeotropic or a critical point,
 * x and y converge (K_i = 1).
 *
 * Distinction between azeotrope and critical point:
 * - At an azeotrope: x = y, but the two-phase region in the PVT isotherm persists.
 *   The K-values cross unity (change from K > 1 to K < 1), and the VLE curve
 *   continues beyond the azeotrope with inverted phase roles.
 * - At a critical point: x = y AND the two-phase region vanishes (inflection point
 *   in the isotherm). The VLE curve terminates here.
 *
 * The function uses the existing samplingForVLE infrastructure, which internally detects
 * the convergence of phase compositions and checks for the existence of a two-phase region
 * to distinguish between the two cases.
 *
 * Assumptions:
 * - Binary mixture only
 * - At least one component must be subcritical at the given temperature (Tc > T)
 * - The VLE region is simply connected (no closed-loop LLE regions interfering)
 *
 * Limitations:
 * - Accuracy is limited by the sampling step size (0.001 in composition)
 * - Systems with both an azeotrope and a critical point at the same T require
 *   tracing through the azeotrope (K-value sign change), which is handled by
 *   checking the two-phase region criterion
 *
 * @param param Global parameters (sw_criticalPoint, sw_azeotropicPoint control behavior)
 * @param calc Calculation object; results are stored in calc.criticalT, calc.criticalP, calc.criticalx
 * @throws std::runtime_error if no critical/azeotropic point is found,
 *         or if the system has more than 2 components
 */
inline void calculateCriticalOrAzeotropicPointForSpecifiedT(parameters& param, calculation& calc) {
    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

        std::vector<Eigen::VectorXd> sampling_x = {};
        std::vector<Eigen::VectorXd> sampling_y = {};
        std::vector<Eigen::VectorXd> sampling_K = {};
        std::vector<Eigen::VectorXd> sampling_P = {};
        std::vector<Eigen::VectorXd> sampling_T = {};

        int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
        double temperature = calc.temperatures[actualConcentrationIndex];

        if (int(calc.components.size() - 1) > 2) {
            throw std::runtime_error("Critical/Azeotropic Point calculation currently only works for binary mixtures.");
        }
        else {
            auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);
            int index = it - calc.uniqueTemperatures.begin();

            // Supercritical / Subcritical detection
            int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
            int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
            auto classification = classifyComponents(subcalcindex_comp1, subcalcindex_comp2, subcalculations[subcalcindex_comp1].criticalT(0), subcalculations[subcalcindex_comp2].criticalT(0), temperature);
            int IndexOfSupercriticalComponent = classification.supercriticalIndex;
            int IndexOfSubcriticalComponent = classification.subcriticalIndex;
            int SubcalculationsIndexOfSupercriticalComponent = classification.supercriticalSubcalcIndex;
            int SubcalculationsIndexOfSubcriticalComponent = classification.subcriticalSubcalcIndex;

            // Tracing starts at sub
            double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
            startx(IndexOfSupercriticalComponent) = 0.0;
            startx(IndexOfSubcriticalComponent) = 1.0;
            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
            startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
            startK(IndexOfSubcriticalComponent) = 1.0;

            std::cout << "Start Sampling For Critical Point Calculation" << std::endl;
            param.sw_Pbub = 1;
            std::vector<std::vector<Eigen::VectorXd>> samplingResults;
            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, 1.0);
            param.sw_Pbub = 0;
            
            sampling_x = samplingResults[0];
            sampling_y = samplingResults[1];
            sampling_K = samplingResults[2];
            sampling_P = samplingResults[3];

            int lastIndex = sampling_x.size() - 1;
            double final_x = sampling_x[lastIndex](IndexOfSupercriticalComponent);
            double final_y = sampling_y[lastIndex](IndexOfSupercriticalComponent);
            double composition_gap = std::abs(final_y - final_x);

            if (final_x > 0.999 || final_x < 0.001 || composition_gap > 0.001) {
                if (param.sw_azeotropicPoint == 1) {
                    throw std::runtime_error("No azeotropic point found for given T. System may be zeotropic.");
                }
                else {
                    throw std::runtime_error("No critical point found for given T. VLE curve ends before reaching criticality.");
                }
            }

            calc.criticalT(i_calculation, 0) = temperature;
            calc.criticalP.row(i_calculation) = sampling_P[lastIndex];
            calc.criticalx.row(i_calculation) = (sampling_y[lastIndex] + sampling_x[lastIndex]) / 2.0;
        }
    }
}


// ====================== ACTIVITY COEFFICIENT FUNCTIONS ======================
/**
 * @brief Computes activity coefficients with openCOSMO-RS-Phi using liquid-like root.
 */
inline void computeActivityCoefficients(parameters& param, calculation& calc) {
    int numberOfOriginalComponents = int(calc.components.size()) - 1;
    calc.overallConcentrations = calc.originalConcentrations.cast<double>();

    calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
    calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;
    
    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
        int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
        double temperature = calc.temperatures[actualConcentrationIndex];
        double pressure = calc.targetPressure(i_calculation);

        ///// Calculate Phi in reference state /////
        Eigen::VectorXd LnPhiRef = Eigen::VectorXd::Zero(int(calc.components.size() - 1));
        for (int ComponentIndex = 0; ComponentIndex < numberOfOriginalComponents; ComponentIndex++) {

            std::vector<double> pressureVector;
            pressureVector.push_back(double(pressure));

            std::vector<std::vector<double>> concentration(1, std::vector<double>(calc.components.size() - 1, 0.0));
            concentration[0][0] = 1.0;

            std::vector<std::shared_ptr<molecule>> componentVector;
            componentVector.push_back(calc.components[ComponentIndex]);
            componentVector.push_back(calc.components[int(calc.components.size() - 1)]); // hole always has to be appended

            calculation calculationScheme = loadSubcalculation(componentVector, { double(temperature) }, concentration, pressureVector);
            int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
            calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});
            calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(1);
            calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(1);
            calculationScheme.mixtureCavityVolume = Eigen::VectorXd::Zero(1);
            calculationScheme.mixtureHoleVolume = Eigen::VectorXd::Zero(1);
            calculationScheme.originalNumberOfCalculations = 2;

            int index = calc.components[ComponentIndex]->index; // Note: index in calculationScheme is different, therefore use _calculation here
            calculationScheme.componentCavityVolume(0) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
            calculationScheme.componentHoleVolume(0) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
            calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
            calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;

            std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 0);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];
            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];

            double v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
            calculationScheme.molarVolume(1) = v;
            calculateEOS(param, calculationScheme, 1);

            LnPhiRef(ComponentIndex) = calculationScheme.lnPhiTotal(1, 0);
        }

        ///// Calculate Fugacity coefficient (liquid-like root) of all components in mixture
        std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
        std::vector<double> liquid_v = returnVector[2];
        std::vector<double> liquid_P = returnVector[3];
        std::vector<double> vapor_v = returnVector[4];
        std::vector<double> vapor_P = returnVector[5];

        double v = calculatevForGivenTPn(calc, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
        calc.molarVolume(i_calculation + numberOfCalculations) = v;
        calculateEOS(param, calc, i_calculation + numberOfCalculations);

        calc.lnGammaTotal.row(i_calculation + numberOfCalculations).head(int(calc.components.size() - 1)) = calc.lnPhiTotal.row(i_calculation + numberOfCalculations) - LnPhiRef.transpose();
    }
}


// ====================== EXCESS AND MIXING PROPERTIES FUNCTIONS ======================
/**
 * @brief Helper function used within calculateExcessAndMixingProperties.
 */
std::vector<Eigen::VectorXd> calculateMolarResidualEnthalpyForPureComponents(calculation _calculation, int i_calculation, int MixtureIndex, std::string mode) {

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[MixtureIndex];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double pressure = _calculation.targetPressure(MixtureIndex);
    double div_Aeff = 1 / param.Aeff;
    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = _calculation.components.size() - 1;
    Eigen::MatrixXd overallConcentrations = _calculation.originalConcentrations.cast<double>();

    Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = Eigen::VectorXd::Zero(_calculation.components.size() - 1);
    Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = Eigen::VectorXd::Zero(_calculation.components.size() - 1);
    Eigen::VectorXd MolarCpresOfPureComponents = Eigen::VectorXd::Zero(_calculation.components.size() - 1);


    auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
    int index = it - _calculation.uniqueTemperatures.begin();

    ///// Reference state / Pure component calculations /////
    for (int i_c = 0; i_c < _calculation.components.size() - 1; i_c++) {

        int subcalcindex = _calculation.IndicesForPureComponentCalculations(index, i_c);

        // Find actual stable state of component i in pure component state for specificied T, P (relevant for h^M calculation)
        int PureCompIndex;
        if (mode == "mixing") {
            if (temperature >= subcalculations[subcalcindex].criticalT(0) || subcalculations[subcalcindex].SaturationPressures(0) < pressure) {
                // This means the pure component is in supercritical or liquid state
                PureCompIndex = 1;
            }
            else {
                // This means the pure component is in gaseous state
                PureCompIndex = 0;
            }
        }
        if (mode == "excess") {
            if (_calculation.calculationType == "LL") {
                PureCompIndex = 1;
            }
            else if (MixtureIndex != i_calculation || temperature >= subcalculations[subcalcindex].criticalT(0)) {
                PureCompIndex = 1;
            }
            else {
                PureCompIndex = 0;
            }
        }
        // Calculate LnGamma = LnPhiMixtureMinusLnPhiPure in case of excess property and LnActivity = Z_i*LnPhiMixture - LnPhiPure in case of mixture property
        LnPhiMixtureMinusLnPhiPure(i_c) = _calculation.lnPhiTotal(MixtureIndex, i_c) - subcalculations[subcalcindex].lnPhiTotal(PureCompIndex, 0);
        if (mode == "mixing" && _calculation.originalConcentrations(MixtureIndex, i_c) != 0.0) {
            double log_xi = log(_calculation.originalConcentrations(MixtureIndex, i_c));
            LnPhiMixtureMinusLnPhiPure(i_c) += log_xi;
        }

        double PureComponentMolarVolume = subcalculations[subcalcindex].molarVolume(PureCompIndex, 0);
        double PureComponentRepulsivePressure = subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0);
        double PureComponentTotalPressure = subcalculations[subcalcindex].totalPressure(PureCompIndex, 0); // should equal target Pressure (of mixture as well as pure component)

        // Calculate h^res(T,v_pure) for current pure component i_c
        int actualConcentrationIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[PureCompIndex];
        int actualReferenceStateIndexPure;
        int numberOfSegments = int(subcalculations[subcalcindex].segments.size());

        double sum_1 = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[PureCompIndex][0]];
            double DivGammaTimesDGamma_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualConcentrationIndexPure));
            double DivGammaIGTimesDGammaIG_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            sum_1 += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][0] * div_Aeff - _calculation.componentCavityVolume(i_c) / _calculation.componentHoleVolume(i_c) * subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (DivGammaTimesDGamma_DT - DivGammaIGTimesDGammaIG_DT);
        }

        double sum_3 = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[PureCompIndex][1]];
            double DivGammaTimesDGammaDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualConcentrationIndexPure));
            double DivGammaIGTimesDGammaIGDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            sum_3 += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT);
        }

        double T2 = temperature * temperature;
        double MinusRT2 = -R_GAS_CONSTANT * T2;
        MolarResidualEnthalpiesOfPureComponents(i_c) = MinusRT2 * sum_1 + MinusRT2 * PureComponentMolarVolume / (_calculation.componentHoleVolume(i_c)) * sum_3 + PureComponentTotalPressure * PureComponentMolarVolume - R_GAS_CONSTANT * temperature;

        // Calculate cp^res(T,v_pure) for current pure component i_c
        MolarCpresOfPureComponents(i_c) = subcalculations[subcalcindex].molarResidualHeatCapacityAtConstantPressure(PureCompIndex);
    }

    std::vector<Eigen::VectorXd> returnVector;
    returnVector.reserve(2);

    returnVector.push_back(LnPhiMixtureMinusLnPhiPure);
    returnVector.push_back(MolarResidualEnthalpiesOfPureComponents);
    returnVector.push_back(MolarCpresOfPureComponents);

    return returnVector;
}


/**
 * @brief Helper function used within calculateExcessAndMixingProperties.
 */
inline void computeExcessForPhase(calculation& calc, int i_calculation, int phaseIndex, int numberOfOriginalComponents, double temperature, double weight)
{
    auto returnVector = calculateMolarResidualEnthalpyForPureComponents(calc, i_calculation, phaseIndex, "excess");
    Eigen::VectorXd LnPhiDiff = returnVector[0];
    Eigen::VectorXd hResPure = returnVector[1];
    Eigen::VectorXd cpResPure = returnVector[2];

    calc.gE(phaseIndex) = weight * R_GAS_CONSTANT * temperature * calc.originalConcentrations.row(phaseIndex).dot(LnPhiDiff);

    Eigen::VectorXd conc = calc.originalConcentrations.row(phaseIndex).head(numberOfOriginalComponents).cast<double>();
    calc.hE(phaseIndex) += -1.0 * conc.dot(hResPure);
    calc.hE(phaseIndex) *= weight;
    calc.cpE(phaseIndex) += -1.0 * conc.dot(cpResPure);
    calc.cpE(phaseIndex) *= weight;
}


/**
 * @brief Helper function used within calculateExcessAndMixingProperties.
 */
inline void computeMixingForPhase(calculation& calc, int i_calculation, int phaseIndex, int numberOfOriginalComponents, double temperature, double weight)
{
    auto returnVector = calculateMolarResidualEnthalpyForPureComponents(calc, i_calculation, phaseIndex, "mixing");
    Eigen::VectorXd LnPhiDiff = returnVector[0];
    Eigen::VectorXd hResPure = returnVector[1];
    Eigen::VectorXd cpResPure = returnVector[2];

    calc.gM(phaseIndex) = weight * R_GAS_CONSTANT * temperature * calc.originalConcentrations.row(phaseIndex).dot(LnPhiDiff);

    Eigen::VectorXd conc = calc.originalConcentrations.row(phaseIndex).head(numberOfOriginalComponents).cast<double>();
    calc.hM(phaseIndex) += -1.0 * conc.dot(hResPure);
    calc.hM(phaseIndex) *= weight;
    calc.cpM(phaseIndex) += -1.0 * conc.dot(cpResPure);
    calc.cpM(phaseIndex) *= weight;
}


/**
 * @brief Calculates excess and mixing properties (g^E, h^E, cp^E, g^M, h^M, cp^M)
 *        for a binary mixture at given (T, P, z) conditions.
 *
 * Algorithm:
 * 1. Phase state determination:
 *    a) If phase boundaries (x_bub, y_dew) for current (T, P) are known from a prior
 *       calculation or external input, the phase state is determined by comparing z
 *       against these boundaries.
 *    b) Otherwise, a PT-Flash is performed (Raoult initialization, then sampling fallback).
 *    c) If no two-phase split is found, the Gibbs energy of both roots (liquid-like,
 *       vapor-like) is compared to determine the stable single phase.
 *
 * 2. Thermodynamic property calculation:
 *    - For single-phase states: residual properties are computed directly from the EOS
 *      at the stable root.
 *    - For two-phase states: residual properties are computed for each phase separately,
 *      then weighted by the phase fraction theta.
 *
 * 3. Excess and mixing property evaluation:
 *    - Excess: h^E = h^res(T,P,z) - sum_i(x_i * h^res_i(T,P_same aggregation state as mixture))
 *    - Mixing: h^M = h^res(T,P,z) - sum_i(x_i * h^res_i(T,P_stable state))
 *    - For two-phase: weighted sum over both phases with theta and (1-theta)
 *
 * Phase boundary caching:
 * Successfully computed phase boundaries are stored in param.temperatures_mixing /
 * param.pressures_mixing / param.concentration_x_phase_mixing / param.concentration_y_phase_mixing
 * so that subsequent calculations at the same (T, P) can skip the Flash.
 *
 * @param param Global parameters; phase boundary cache is modified as side effect
 * @param calc Calculation object; results stored in calc.gE, calc.hE, calc.cpE, calc.gM, calc.hM, calc.cpM
 */
inline void calculateExcessAndMixingProperties(parameters& param, calculation& calc) {

    calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
    calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

    int numberOfCalculations = int(calc.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = int(calc.components.size()) - 1;

    std::vector<Eigen::VectorXd> sampling_x = {};
    std::vector<Eigen::VectorXd> sampling_y = {};
    std::vector<Eigen::VectorXd> sampling_K = {};
    std::vector<Eigen::VectorXd> sampling_P = {};
    std::vector<Eigen::VectorXd> sampling_T = {};

    for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

        int actualConcentrationIndex = calc.actualConcentrationIndices[i_calculation];
        double temperature = calc.temperatures[actualConcentrationIndex];
        double pressure = calc.targetPressure(i_calculation);

        bool phaseBoundariesKnown = false;

        // Check if phase boundaries for current T, P are given exernally 
        int indexInExternalParamVectors = -1;
        for (size_t i = 0; i < param.temperatures_mixing.size(); ++i) {
            if (std::abs(param.temperatures_mixing[i] - temperature) <= 1e-6 && std::abs(param.pressures_mixing[i] - pressure) <= 1e-6) {
                phaseBoundariesKnown = true;
                indexInExternalParamVectors = i;
                break;
            }
        }

        std::string MixtureState = "UNKNOWN";

        // If phase boundaries are given externally for current T, P, actual mixture state is directly known 
        if (phaseBoundariesKnown) {
            bool all_zero = std::all_of(param.concentration_x_phase_mixing[indexInExternalParamVectors].begin(), param.concentration_x_phase_mixing[indexInExternalParamVectors].end(), [](double elem) {
                return elem == 0.0;
                });
            bool all_one = std::all_of(param.concentration_x_phase_mixing[indexInExternalParamVectors].begin(), param.concentration_x_phase_mixing[indexInExternalParamVectors].end(), [](double elem) {
                return elem == 1.0;
                });

            if (all_zero || (calc.overallConcentrations(i_calculation, 0) <= param.concentration_x_phase_mixing[indexInExternalParamVectors][0])) {
                MixtureState = "liquid";
            }
            else if ((param.concentration_x_phase_mixing[indexInExternalParamVectors][0] < calc.overallConcentrations(i_calculation, 0)) && (calc.overallConcentrations(i_calculation, 0) < param.concentration_y_phase_mixing[indexInExternalParamVectors][0])) {
                MixtureState = "TwoPhaseSplit";
                for (int m = 0; m < calc.components.size() - 1; m++) {
                    calc.originalConcentrations(i_calculation + numberOfCalculations, m) = param.concentration_x_phase_mixing[indexInExternalParamVectors][m];
                    calc.originalConcentrations(i_calculation, m) = param.concentration_y_phase_mixing[indexInExternalParamVectors][m];
                }
            }
            else if (all_one || (calc.overallConcentrations(i_calculation, 0) >= param.concentration_y_phase_mixing[indexInExternalParamVectors][0])) {
                MixtureState = "gaseous";
            }
        }

        // If phase boundaries are not given externally for current T, P, actual mixture state has to be determined by performing a pT-Flash
        else if (!(phaseBoundariesKnown)) {
            double thetaGuess = 0.0;
            bool skipThetaCalculation = false;
            bool skipFlashCalculation = false;
            auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);
            int index = it - calc.uniqueTemperatures.begin();
            int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
            int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
            if (subcalculations[subcalcindex_comp1].criticalT(0) < temperature && subcalculations[subcalcindex_comp2].criticalT(0) < temperature) {
                skipFlashCalculation = true;
                MixtureState = "UNKNOWN";
            }

            // Initialization Routine 1: Shortcut K-ratio method / Raoult's law
            if (calc.calculationType == "LL") {
                // Initialization scheme for LLE
                thetaGuess = initializeLLEFlash(param, calc, i_calculation, thetaGuess, skipThetaCalculation);
                skipThetaCalculation = true;

                calculatePTFlash(calc, i_calculation, 0.001, skipThetaCalculation, thetaGuess, true);

                if (!(calc.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(calc.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(calc.totalPressure(i_calculation) == 0.0)) {
                    param.temperatures_mixing.push_back(temperature);
                    param.pressures_mixing.push_back(pressure);
                    std::vector<double> x_vector = {};
                    std::vector<double> y_vector = {};
                    for (size_t i_c = 0; i_c + 1 < calc.components.size(); ++i_c) {
                        x_vector.push_back(calc.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                        y_vector.push_back(calc.originalConcentrations(i_calculation, i_c));
                    }
                    param.concentration_x_phase_mixing.push_back(x_vector);
                    param.concentration_y_phase_mixing.push_back(y_vector);
                    MixtureState = "TwoPhaseSplit";
                }
            }

            else if (calc.calculationType == "VL") {
                if (skipFlashCalculation == false && sampling_x.empty()) {
                    calculatePTFlash(calc, i_calculation, 0.001, skipThetaCalculation, thetaGuess);
                    if (!(calc.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(calc.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(calc.totalPressure(i_calculation) == 0.0)) {
                        param.temperatures_mixing.push_back(temperature);
                        param.pressures_mixing.push_back(pressure);
                        std::vector<double> x_vector = {};
                        std::vector<double> y_vector = {};
                        for (size_t i_c = 0; i_c + 1 < calc.components.size(); ++i_c) {
                            x_vector.push_back(calc.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                            y_vector.push_back(calc.originalConcentrations(i_calculation, i_c));
                        }
                        param.concentration_x_phase_mixing.push_back(x_vector);
                        param.concentration_y_phase_mixing.push_back(y_vector);
                        MixtureState = "TwoPhaseSplit";
                    }
                }

                // Attempt 2: Sampling-based initialization
                if (skipFlashCalculation == false && (calc.totalPressure(i_calculation) == 0.0 || sampling_x.size() > 0)) {
                    // In this case no equilibrium was found

                    auto it = std::find(calc.uniqueTemperatures.begin(), calc.uniqueTemperatures.end(), temperature);
                    int index = it - calc.uniqueTemperatures.begin();

                    int subcalcindex_comp1 = calc.IndicesForPureComponentCalculations(index, 0);
                    int subcalcindex_comp2 = calc.IndicesForPureComponentCalculations(index, 1);
                    auto classification = classifyComponents(subcalcindex_comp1, subcalcindex_comp2, subcalculations[subcalcindex_comp1].criticalT(0), subcalculations[subcalcindex_comp2].criticalT(0), temperature, false);
                    int IndexOfSupercriticalComponent = classification.supercriticalIndex;
                    int IndexOfSubcriticalComponent = classification.subcriticalIndex;
                    int SubcalculationsIndexOfSupercriticalComponent = classification.supercriticalSubcalcIndex;
                    int SubcalculationsIndexOfSubcriticalComponent = classification.subcriticalSubcalcIndex;

                    if (IndexOfSupercriticalComponent == 1 && IndexOfSubcriticalComponent == 1) {
                        skipFlashCalculation = true;
                    }

                    double endx = calc.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

                    if (skipFlashCalculation == false && sampling_x.empty()) {

                        if (int(calc.components.size() - 1) > 2) {
                            throw std::runtime_error("PT Flash Algorithm did not converge. Sampling currently only works for binary mixtures.");
                        }
                        else {
                            std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                            param.sw_Pbub = 1;
                            double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
                            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                            startx(IndexOfSupercriticalComponent) = 0.0;
                            startx(IndexOfSubcriticalComponent) = 1.0;
                            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
                            startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
                            startK(IndexOfSubcriticalComponent) = 1.0;
                            samplingResults = samplingForVLE(calc.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
                            sampling_x = samplingResults[0];
                            sampling_y = samplingResults[1];
                            sampling_K = samplingResults[2];
                            sampling_P = samplingResults[3];
                            param.sw_Pbub = 0;
                        }
                    }

                    auto range = determineSamplingRange(sampling_x, sampling_y, sampling_P, IndexOfSupercriticalComponent);

                    double z = calc.overallConcentrations(i_calculation, IndexOfSupercriticalComponent);
                    double P_target = calc.targetPressure(i_calculation);

                    auto searchResult = findTwoPhaseSegment(sampling_x, sampling_y, sampling_P,
                        IndexOfSupercriticalComponent, z, P_target, range);

                    if (searchResult.twoPhaseFound) {
                        // Setze Startwerte aus Sampling und berechne Flash
                        calc.originalConcentrations.row(i_calculation) = sampling_y[searchResult.twoPhaseIndex];
                        calc.originalConcentrations.row(i_calculation + numberOfCalculations) = sampling_x[searchResult.twoPhaseIndex];
                        calc.PartitionCoefficients.row(i_calculation) = sampling_K[searchResult.twoPhaseIndex];
                        calculatePTFlash(calc, i_calculation, 0.001, false, 0.0, true);

                        if (!(calc.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(calc.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(calc.totalPressure(i_calculation) == 0.0)) {
                            param.temperatures_mixing.push_back(temperature);
                            param.pressures_mixing.push_back(pressure);
                            std::vector<double> x_vector = {};
                            std::vector<double> y_vector = {};
                            for (size_t i_c = 0; i_c + 1 < calc.components.size(); ++i_c) {
                                x_vector.push_back(calc.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                                y_vector.push_back(calc.originalConcentrations(i_calculation, i_c));
                            }
                            param.concentration_x_phase_mixing.push_back(x_vector);
                            param.concentration_y_phase_mixing.push_back(y_vector);
                            MixtureState = "TwoPhaseSplit";
                        }
                    }
                    else if (searchResult.zInAnySegment) {
                        if (P_target < searchResult.P_low) {
                            MixtureState = "gaseous";
                        }
                        else if (P_target > searchResult.P_high) {
                            MixtureState = "liquid";
                        }
                    }
                }
            }
        }


        // Indices check
        int MixtureIndex = -1;
        double theta = 0.0; // molar proportion of the gas phase
        double RT = R_GAS_CONSTANT * temperature;
        if (MixtureState == "UNKNOWN") {
            // This means that no bubble and/or dew point could be calculated in Flash procedure
            // First calculate residual molar gibbs energy selecting the liquid-like root

            calc.originalConcentrations.row(i_calculation) = calc.overallConcentrations.row(i_calculation);
            calc.originalConcentrations.row(i_calculation + numberOfCalculations) = calc.overallConcentrations.row(i_calculation);

            calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
            calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

            std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            double v = calculatevForGivenTPn(calc, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
            double residualGibbsEnergyLiquid = 1e8;
            if (v != 1000) {
                residualGibbsEnergyLiquid = 0.0;
                calc.molarVolume(i_calculation + numberOfCalculations) = v;
                calculateEOS(param, calc, i_calculation + numberOfCalculations, true, false);

                for (size_t i_c = 0; i_c + 1 < calc.components.size(); ++i_c) {
                    residualGibbsEnergyLiquid += calc.originalConcentrations(i_calculation + numberOfCalculations, i_c) * RT * (calc.lnGammaResidual(i_calculation + numberOfCalculations, i_c) - calc.componentCavityVolume(i_c) / calc.mixtureHoleVolume(i_calculation + numberOfCalculations) * calc.lnGammaResidual(i_calculation + numberOfCalculations, calc.components.size() - 1));
                }
                double z_liquid = calc.totalPressure(i_calculation + numberOfCalculations) * v / RT;
                residualGibbsEnergyLiquid = residualGibbsEnergyLiquid - calc.attractivePressure(i_calculation + numberOfCalculations) * v - RT * log((v - calc.mixtureCavityVolume(i_calculation + numberOfCalculations)) / v) + RT * (z_liquid - 1) - RT * log(z_liquid);
            }

            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];

            if (vapor_P.size() > 0 && pressure <= vapor_P[0]) {
                double residualGibbsEnergyVapor = 1e8;
                v = calculatevForGivenTPn(calc, i_calculation, vapor_v, vapor_P, pressure);
                if (v != 1000) {
                    residualGibbsEnergyVapor = 0.0;
                    calc.molarVolume(i_calculation) = v;
                    calculateEOS(param, calc, i_calculation, true, false);

                    for (size_t i_c = 0; i_c + 1 < calc.components.size(); ++i_c) {
                        residualGibbsEnergyVapor += calc.originalConcentrations(i_calculation, i_c) * RT * (calc.lnGammaResidual(i_calculation, i_c) - calc.componentCavityVolume(i_c) / calc.mixtureHoleVolume(i_calculation) * calc.lnGammaResidual(i_calculation, calc.components.size() - 1));
                    }
                    double z_vapor = calc.totalPressure(i_calculation) * v / RT;
                    residualGibbsEnergyVapor = residualGibbsEnergyVapor - calc.attractivePressure(i_calculation) * v - RT * log((v - calc.mixtureCavityVolume(i_calculation)) / v) + RT * (z_vapor - 1) - RT * log(z_vapor);
                }

                if (residualGibbsEnergyVapor < residualGibbsEnergyLiquid) {
                    MixtureState = "gaseous";
                    theta = 1;
                    MixtureIndex = i_calculation;
                }
                else {
                    MixtureState = "liquid";
                    theta = 0;
                    MixtureIndex = i_calculation + numberOfCalculations;
                }
            }
            else {
                MixtureState = "liquid";
                theta = 0;
                MixtureIndex = i_calculation + numberOfCalculations;
            }
        }
        else if (MixtureState == "liquid") {
            theta = 0.0;
            calc.originalConcentrations.row(i_calculation + numberOfCalculations) = calc.overallConcentrations.row(i_calculation);
            calc.originalConcentrations.row(i_calculation).setZero();
            MixtureIndex = i_calculation + numberOfCalculations;
        }
        else if (MixtureState == "gaseous") {
            theta = 1.0;
            calc.originalConcentrations.row(i_calculation) = calc.overallConcentrations.row(i_calculation);
            calc.originalConcentrations.row(i_calculation + numberOfCalculations).setZero();
            MixtureIndex = i_calculation;
        }
        else if (MixtureState == "TwoPhaseSplit") {
            // Mixture splits into two phases
            theta = std::abs((calc.overallConcentrations(i_calculation, 0) - calc.originalConcentrations(i_calculation + numberOfCalculations, 0)) / (calc.originalConcentrations(i_calculation, 0) - calc.originalConcentrations(i_calculation + numberOfCalculations, 0)));
        }

        calc.mixtureCavityVolume = calc.originalConcentrations.cast<double>() * calc.componentCavityVolume;
        calc.mixtureHoleVolume = calc.originalConcentrations.cast<double>() * calc.componentHoleVolume;

        if (MixtureState == "liquid" || MixtureState == "TwoPhaseSplit") {
            std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            double v = calculatevForGivenTPn(calc, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
            if (v == 1000.0) {
                liquid_v = returnVector[4];
                liquid_P = returnVector[5];
                v = calculatevForGivenTPn(calc, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
            }
            calc.molarVolume(i_calculation + numberOfCalculations) = v;
            calculateEOS(param, calc, i_calculation + numberOfCalculations, true, true);

            // Save h^res(T,v,z) for mixture in hM
            calc.hM(i_calculation + numberOfCalculations) = calc.molarResidualEnthalpy(i_calculation + numberOfCalculations);
            calc.hE(i_calculation + numberOfCalculations) = calc.hM(i_calculation + numberOfCalculations);
            calc.cpM(i_calculation + numberOfCalculations) = calc.molarResidualHeatCapacityAtConstantPressure(i_calculation + numberOfCalculations);
            calc.cpE(i_calculation + numberOfCalculations) = calc.cpM(i_calculation + numberOfCalculations);
        }

        if (MixtureState == "gaseous" || MixtureState == "TwoPhaseSplit") {
            std::vector<std::vector<double>> returnVector = calculateIsotherm(calc, i_calculation);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];
            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];
            if (calc.calculationType == "LL") {
                // In the case of a (second) liquid phase, the v root should normally be found in returnVector[2] and returnVector[4] is fallback.
                vapor_v = returnVector[2];
                vapor_P = returnVector[3];
                liquid_v = returnVector[4];
                liquid_P = returnVector[5];
            }
            double v = calculatevForGivenTPn(calc, i_calculation, vapor_v, vapor_P, pressure);
            if (v == 1000.0) {
                v = calculatevForGivenTPn(calc, i_calculation, liquid_v, liquid_P, pressure);
            }
            calc.molarVolume(i_calculation) = v;
            calculateEOS(param, calc, i_calculation, true, true);

            // Initially set mixture properties equal to residual property of mixture (later arithmetic mean of pc properties is subtracted)
            calc.hM(i_calculation) = calc.molarResidualEnthalpy(i_calculation);
            calc.hE(i_calculation) = calc.hM(i_calculation);
            calc.cpM(i_calculation) = calc.molarResidualHeatCapacityAtConstantPressure(i_calculation);
            calc.cpE(i_calculation) = calc.cpM(i_calculation);
        }

        // Excess and Mixing Properties // 
        if (MixtureState != "TwoPhaseSplit") {
            computeExcessForPhase(calc, i_calculation, MixtureIndex, numberOfOriginalComponents, temperature, 1.0);
            computeMixingForPhase(calc, i_calculation, MixtureIndex, numberOfOriginalComponents, temperature, 1.0);
        }
        else {
            computeExcessForPhase(calc, i_calculation, i_calculation, numberOfOriginalComponents, temperature, theta);
            computeExcessForPhase(calc, i_calculation, i_calculation + numberOfCalculations, numberOfOriginalComponents, temperature, 1.0 - theta);
            computeMixingForPhase(calc, i_calculation, i_calculation, numberOfOriginalComponents, temperature, theta);
            computeMixingForPhase(calc, i_calculation, i_calculation + numberOfCalculations, numberOfOriginalComponents, temperature, 1.0 - theta);
        }
    }
}


// ====================== SUBCALCULATION FUNCTIONS ======================
/**
 * @brief This function is built to perform the reference / pure component state calculations needed to calculate 
 * phase equilibrium properties or calculations that are created within the code and not specified externally by the user.
 * It is structurally quite similar to the calculate function but works with the subcalculations vector.
 */
inline void calculateSubcalculations(std::vector<int>& calculationIndices) {

    // this is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section ends
    threadException e;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < calculationIndices.size(); i++) {

        e.run([=] {
            int calculationIndex = calculationIndices[i];

#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
            if (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) {

                subcalculations[calculationIndex].segments.clear();

                for (int j = 0; j < subcalculations[calculationIndex].components.size(); j++) {

                    std::shared_ptr<molecule> thisMolecule = subcalculations[calculationIndex].components[j];

                    for (int k = 0; k < thisMolecule->segments.size(); k++) {
                        subcalculations[calculationIndex].segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
                            thisMolecule->segments.SegmentTypeSigma[k],
                            thisMolecule->segments.SegmentTypeSigmaCorr[k],
                            thisMolecule->segments.SegmentTypeHBtype[k],
                            thisMolecule->segments.SegmentTypeAtomicNumber[k],
                            thisMolecule->segments.SegmentTypeAreas[k][0],
                            thisMolecule->segments.SegmentTypeMoleculeIndex[k]);
                    }
                }
                subcalculations[calculationIndex].segments.sort();
                subcalculations[calculationIndex].segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasSecondPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()));
                subcalculations[calculationIndex].PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()));
            }


            if (param.sw_alwaysCalculateSizeRelatedParameters == 1 || (param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) \
                || (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {

                rescaleSegments(param, subcalculations[calculationIndex]);
                calculateSegmentConcentrations(subcalculations[calculationIndex]);
            }

#ifdef DEBUG_INFO
            display("number of components: " + std::to_string(calculations[calculationIndex].components.size()) + "\n");
            display("number of segments: " + std::to_string(calculations[calculationIndex].segments.size()) + "\n");
            display("number of concentrations: " + std::to_string(calculations[calculationIndex].concentrations.size()) + "\n");
#endif

#ifdef MEASURE_TIME
            rescaleSegments_total_ms += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count();
            std::chrono::high_resolution_clock::time_point calculateCombinatorial_last = std::chrono::high_resolution_clock::now();
#endif

            if (param.sw_phi == 1) {

                calculation _calculation = subcalculations[calculationIndex];

                _calculation.originalConcentrations = subcalculations[calculationIndex].originalConcentrations; // for excess properties currently necessary because data behind maps are lost

                int numberOfOriginalComponents = _calculation.components.size() - 1;

                _calculation.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);

                for (int i = 0; i < numberOfOriginalComponents; i++) {
                    int index = _calculation.components[i]->index;
                    _calculation.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                    _calculation.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                }

                if (numberOfOriginalComponents == 1) {

                    for (int j = 0; j < _calculation.originalConcentrations.rows(); j++) {
                        _calculation.originalConcentrations(j, 0) = 1.0;
                    }

                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                    if (_calculation.originalNumberOfCalculations % 2 != 0)
                        throw std::runtime_error("Both phases are expected to be given as concentrations.");

                    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);
                    _calculation.SaturationPressures = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), 1);

                    //#if defined(_OPENMP)
                    //#pragma omp parallel for
                    //#endif
                    for (int i_calculation = 0; i_calculation < numberOfStatePoints; i_calculation++) {

                        findCriticalPointForSpecifiedx(_calculation, i_calculation);

                        double endP = 10.0;
                        if (param.sw_excess == 1) {
                            endP = 9.99;
                        }

                        std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation, endP);

                        std::vector<double> v = returnVector[0];
                        std::vector<double> P = returnVector[1];
                        std::vector<double> liquid_v = returnVector[2];
                        std::vector<double> liquid_P = returnVector[3];
                        std::vector<double> vapor_v = returnVector[4];
                        std::vector<double> vapor_P = returnVector[5];
                        std::vector<double> P_repulsive = returnVector[6];
                        std::vector<double> P_attractive = returnVector[7];

                        if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_excess == 1) {
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfStatePoints];
                            int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            bool calculatePsatWithWilson = false;
                            if (vapor_v.size() == 0 || vapor_v.size() == 1 || temperature > _calculation.criticalT(i_calculation)) {
                                // In this case the given temperature is greater than the component's critical temperature. Then, P_isat should be calculated with Wilson Correlation.
                                calculatePsatWithWilson = true;
                            }
                            else {
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                if (_calculation.totalPressure(i_calculation) == 0.0) {
                                    calculatePsatWithWilson = true;
                                }
                            }
                            if (calculatePsatWithWilson == true) {
                                double temperatureNeededForAcentricFactor = 0.7 * _calculation.criticalT(i_calculation);
                                _calculation.temperatures[actualConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualConcentrationIndex2] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.TauTemperatures[0] = temperatureNeededForAcentricFactor;
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                std::vector<double> v = returnVector[0];
                                std::vector<double> P = returnVector[1];
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];
                                std::vector<double> P_repulsive = returnVector[6];
                                std::vector<double> P_attractive = returnVector[7];
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                double reducedSaturationVaporPressure = _calculation.totalPressure(i_calculation) / _calculation.criticalP(i_calculation);
                                double acentricFactor = -1 * log10(reducedSaturationVaporPressure) - 1;
                                // Calculate Saturation Pressure with Wilson Correlation and save it in totalPressure
                                _calculation.totalPressure(i_calculation) = _calculation.criticalP(i_calculation) * exp(5.37 * (1 + acentricFactor) * (1 - _calculation.criticalT(i_calculation) / temperature));
                                _calculation.totalPressure(i_calculation + numberOfStatePoints) = _calculation.criticalP(i_calculation) * exp(5.37 * (1 + acentricFactor) * (1 - _calculation.criticalT(i_calculation) / temperature));
                                // This is just to restore original temperature in the temperature vectors
                                _calculation.temperatures[actualConcentrationIndex] = temperature;
                                _calculation.temperatures[actualConcentrationIndex2] = temperature;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperature;
                                _calculation.TauTemperatures[0] = temperature;
                            }
                            _calculation.SaturationPressures(i_calculation, 0) = _calculation.totalPressure(i_calculation);
                            subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;

                            if (param.sw_excess == 1) {
                                double pressure = _calculation.targetPressure(i_calculation);
                                double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, liquid_v, liquid_P, pressure);
                                if (v == 1000.0) {
                                    v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, vapor_v, vapor_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation + numberOfStatePoints) = v;
                                calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, true, true);

                                v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                                if (v == 1000.0) {
                                    v = calculatevForGivenTPn(_calculation, i_calculation, liquid_v, liquid_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation) = v;
                                calculateEOS(param, _calculation, i_calculation, true, true);

                                subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;
                                subcalculations[calculationIndex].segmentGammas = _calculation.segmentGammas;
                                subcalculations[calculationIndex].segmentGammasPDETemperature = _calculation.segmentGammasPDETemperature;
                                subcalculations[calculationIndex].segmentGammasPDEVolume = _calculation.segmentGammasPDEVolume;
                                subcalculations[calculationIndex].segmentGammasSecondPDETemperature = _calculation.segmentGammasSecondPDETemperature;
                                subcalculations[calculationIndex].repulsivePressure = _calculation.repulsivePressure;
                                subcalculations[calculationIndex].molarResidualEnthalpy = _calculation.molarResidualEnthalpy;
                                subcalculations[calculationIndex].molarResidualHeatCapacityAtConstantPressure = _calculation.molarResidualHeatCapacityAtConstantPressure;
                                //subcalculations[calculationIndex] = _calculation;
                            }
                        }
                        else if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
                            double pr = _calculation.targetPressure(i_calculation) / _calculation.criticalP(i_calculation);  // Reduced pressure
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfStatePoints];
                            int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                            if (pr > 1) {
                                double temperatureNeededForAcentricFactor = 0.7 * _calculation.criticalT(i_calculation);
                                _calculation.temperatures[actualConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualConcentrationIndex2] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.TauTemperatures[0] = temperatureNeededForAcentricFactor;
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                std::vector<double> v = returnVector[0];
                                std::vector<double> P = returnVector[1];
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];
                                std::vector<double> P_repulsive = returnVector[6];
                                std::vector<double> P_attractive = returnVector[7];
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                double reducedSaturationVaporPressure = _calculation.totalPressure(i_calculation) / _calculation.criticalP(i_calculation);
                                double acentricFactor = -1 * log10(reducedSaturationVaporPressure) - 1;
                                double TsatWithWilson = _calculation.criticalT(0) / (1 - 1 / (5.373 * (1 + acentricFactor)) * log(pr));
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex] = TsatWithWilson;
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex2] = TsatWithWilson;
                                subcalculations[calculationIndex].temperatures[actualReferenceStateConcentrationIndex] = TsatWithWilson;
                                subcalculations[calculationIndex].TauTemperatures[0] = TsatWithWilson;
                            }
                            else {
                                std::vector<double> TemperatureGuessesTried = std::vector<double>();
                                int i_calculation_vapor = i_calculation;
                                int i_calculation_liquid = i_calculation + numberOfStatePoints;
                                double TemperatureGuess = _calculation.temperatures[actualConcentrationIndex];

                                double Tmax = 0.999 * _calculation.criticalT(i_calculation);

                                if (pr < 0.2) {
                                    TemperatureGuess = 0.3 * _calculation.criticalT(i_calculation);
                                }
                                else if (pr < 0.8) {
                                    TemperatureGuess = 0.5 * _calculation.criticalT(i_calculation);
                                }
                                else {
                                    TemperatureGuess = 0.85 * _calculation.criticalT(i_calculation);
                                }
                                double stepsize = 0.01 * abs(Tmax - TemperatureGuess);

                                double TemperatureLimit1 = TemperatureGuess;
                                double TemperatureLimit2;
                                bool TemperatureLimit2Found = false;
                                double OF_last = 0;
                                double OF_Limit1 = 0;
                                double OF_Limit2 = 0;
                                std::vector<std::vector<double>> returnVector;
                                std::vector<double> v;
                                std::vector<double> P;
                                std::vector<double> liquid_v;
                                std::vector<double> liquid_P;
                                std::vector<double> vapor_v;
                                std::vector<double> vapor_P;
                                std::vector<double> P_repulsive;
                                std::vector<double> P_attractive;

                                while (TemperatureLimit2Found == false) {
                                    _calculation.temperatures[actualConcentrationIndex] = TemperatureGuess;
                                    _calculation.temperatures[actualConcentrationIndex2] = TemperatureGuess;
                                    _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
                                    _calculation.TauTemperatures[0] = TemperatureGuess;
                                    int size = 0;

                                    while (size < 2) {
                                        returnVector = calculateIsotherm(_calculation, i_calculation);
                                        v = returnVector[0];
                                        P = returnVector[1];
                                        liquid_v = returnVector[2];
                                        liquid_P = returnVector[3];
                                        vapor_v = returnVector[4];
                                        vapor_P = returnVector[5];
                                        P_repulsive = returnVector[6];
                                        P_attractive = returnVector[7];
                                        size = int(vapor_v.size());
                                        if (size == 0 || size == 1) {
                                            TemperatureGuess = TemperatureGuess - 10;
                                        }
                                        size = int(liquid_v.size());
                                        if (size == 1) {
                                            TemperatureGuess = TemperatureGuess + stepsize;
                                            if (TemperatureGuess > Tmax) {
                                                TemperatureGuess = Tmax;
                                            }
                                        }
                                    }

                                    calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation_vapor, 0));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation_liquid, 0));
                                    double OF_T = fugacityCoefficientVapor - fugacityCoefficientLiquid;
                                    if (OF_last * OF_T < 0) {
                                        TemperatureLimit2 = TemperatureGuess;
                                        OF_Limit2 = OF_T;
                                        TemperatureLimit2Found = true;
                                    }
                                    else {
                                        TemperatureLimit1 = TemperatureGuess;
                                        OF_Limit1 = OF_T;
                                    }
                                    OF_last = OF_T;
                                    TemperatureGuess += stepsize;
                                    if (TemperatureGuess > Tmax) {
                                        TemperatureGuess = Tmax;
                                    }
                                }


                                auto f_temperature = [&_calculation, i_calculation, actualConcentrationIndex, actualConcentrationIndex2, actualReferenceStateConcentrationIndex](double temperature) {
                                    _calculation.temperatures[actualConcentrationIndex] = temperature;
                                    _calculation.temperatures[actualConcentrationIndex2] = temperature;
                                    _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperature;
                                    _calculation.TauTemperatures[0] = temperature;

                                    std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                    std::vector<double> v = returnVector[0];
                                    std::vector<double> P = returnVector[1];
                                    std::vector<double> liquid_v = returnVector[2];
                                    std::vector<double> liquid_P = returnVector[3];
                                    std::vector<double> vapor_v = returnVector[4];
                                    std::vector<double> vapor_P = returnVector[5];
                                    std::vector<double> P_repulsive = returnVector[6];
                                    std::vector<double> P_attractive = returnVector[7];

                                    calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, 0));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + int(_calculation.originalNumberOfCalculations / 2), 0));
                                    double OF_T = fugacityCoefficientVapor - fugacityCoefficientLiquid;

                                    return OF_T;
                                    };

                                double temperatureGuess = ITP(f_temperature, TemperatureLimit1, TemperatureLimit2, OF_Limit1, OF_Limit2, 1e-10, 1e-10);

                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex] = temperatureGuess;
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex2] = temperatureGuess;
                                subcalculations[calculationIndex].temperatures[actualReferenceStateConcentrationIndex] = temperatureGuess;
                                subcalculations[calculationIndex].TauTemperatures[0] = temperatureGuess;
                                //calculateEOS(param, subcalculations[calculationIndex], i_calculation);
                            }
                        }
                        else {
                            _calculation.SaturationPressures(i_calculation, 0) = _calculation.totalPressure(i_calculation);

                            double pressure = _calculation.targetPressure(i_calculation);
                            double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, liquid_v, liquid_P, pressure);
                            _calculation.molarVolume(i_calculation + numberOfStatePoints) = v;
                            calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, true);

                            v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                            _calculation.molarVolume(i_calculation) = v;
                            calculateEOS(param, _calculation, i_calculation, true);
                            subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;
                            subcalculations[calculationIndex].segmentGammas = _calculation.segmentGammas;
                            subcalculations[calculationIndex].segmentGammasPDETemperature = _calculation.segmentGammasPDETemperature;
                            subcalculations[calculationIndex].segmentGammasPDEVolume = _calculation.segmentGammasPDEVolume;
                            subcalculations[calculationIndex].segmentGammasSecondPDETemperature = _calculation.segmentGammasSecondPDETemperature;
                            subcalculations[calculationIndex].repulsivePressure = _calculation.repulsivePressure;
                            //subcalculations[calculationIndex] = _calculation;

                        }
                    }
                }
            }
            });
    }
    e.rethrow();
}
/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once

#include "interaction_matrix.hpp"
#include "contact_statistics.hpp"
#include "COSMOfile_functions.hpp"
#include "simd.hpp"
#include <stdexcept>
#include <functional>
#include <cmath>
#include <limits>


#if defined(MEASURE_TIME) 
#include <chrono>

std::atomic<unsigned long> oneIteration_total_ms = 0;
std::chrono::high_resolution_clock::time_point oneIteration_last;

std::atomic<unsigned long> calculateTau_total_ms = 0;
std::atomic<unsigned long> calculateCOSMOSPACE_total_ms = 0;
std::atomic<unsigned long> calculateGammasForMolecules_total_ms = 0;
std::atomic<unsigned long> calculateContactStatistics_total_ms = 0;
std::atomic<unsigned long> rescaleSegments_total_ms = 0;
std::atomic<unsigned long> calculateCombinatorial_total_ms = 0;
std::atomic<unsigned long> calculateResidual_total_ms = 0;
std::atomic<unsigned long> addContributions_total_ms = 0;

void startCalculationMeasurement() {

    oneIteration_last = std::chrono::high_resolution_clock::now();

    // by commenting the following lines you achieve a cummulative sum of the time over more than one iteration.
    rescaleSegments_total_ms = 0;
    calculateCombinatorial_total_ms = 0;
    calculateResidual_total_ms = 0;
    calculateTau_total_ms = 0;
    calculateCOSMOSPACE_total_ms = 0;
    calculateContactStatistics_total_ms = 0;
    calculateGammasForMolecules_total_ms = 0;
    addContributions_total_ms = 0;
    oneIteration_total_ms = 0;
}

void stopCalculationMeasurement() {

    oneIteration_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - oneIteration_last).count());

    // print accumulated times
    displayTime("rescaleSegments_total_ms               ", rescaleSegments_total_ms);
    displayTime("calculateCombinatorial_total_ms        ", calculateCombinatorial_total_ms);
    displayTime("calculateResidual_total_ms             ", calculateResidual_total_ms);
    displayTime("   calculateTau_total_ms               ", calculateTau_total_ms);
    displayTime("   calculateCOSMOSPACE_total_ms        ", calculateCOSMOSPACE_total_ms);
    if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
        displayTime("   calculateContactStatistics_total_ms ", calculateContactStatistics_total_ms);
    }
    displayTime("   calculateGammasForMolecules_total_ms", calculateGammasForMolecules_total_ms);
    displayTime("addContributions_total_ms              ", addContributions_total_ms);
    displayTime("oneIteration_total_ms                  ", oneIteration_total_ms);
    display("\n");
}
#endif

void initialize(parameters& param, bool initializeParameters = true, bool initializeMolecules = true, bool initializeCalculations = true, bool showBinarySpecs = true) {


    if (initializeParameters) {
        n_ex = 0;
        param.ChargeRaster.clear();
        param.exp_param.clear();
        param.R_i = std::vector<double>(118, 0.0);
        param.R_i_COSMO = std::vector<double>(118, 0.0);
        param.HBClassElmnt = std::vector<int>(300, 0);

        // Initialize hydrogen bond classes of the elements HBClassElmnt
        // 0 : only non HB  | 1 : potential donor  | 2 : potential acceptor | 3 : potential donor or acceptor
        // classify all hydrogens and some metals as potential donors and all others as potential acceptors.

        for (int atomic_number = 0; atomic_number < param.HBClassElmnt.size(); atomic_number++) {
            if (atomic_number <= 100) param.HBClassElmnt[atomic_number] = 2;
            else if (atomic_number > 100) param.HBClassElmnt[atomic_number] = 1;   // all hydrogens
        }

        // set some values manually
        param.HBClassElmnt[1] = 1;   // hydrogen
        param.HBClassElmnt[3] = 1;   // li 
        param.HBClassElmnt[4] = 1;   // be 
        param.HBClassElmnt[11] = 1;  // na 
        param.HBClassElmnt[12] = 1;  // mg 
        param.HBClassElmnt[13] = 1;  // al 
        param.HBClassElmnt[19] = 1;  // k* 
        param.HBClassElmnt[20] = 1;  // ca 
        param.HBClassElmnt[24] = 1;  // cr 
        param.HBClassElmnt[26] = 1;  // fe 
        param.HBClassElmnt[27] = 1;  // co 
        param.HBClassElmnt[29] = 1;  // cu 
        param.HBClassElmnt[30] = 1;  // zn 
        param.HBClassElmnt[37] = 1;  // rb 
        param.HBClassElmnt[38] = 1;  // sr 
        param.HBClassElmnt[48] = 1;  // cd 
        param.HBClassElmnt[55] = 1;  // cs 
        param.HBClassElmnt[56] = 1;  // ba 


        // initialize charge raster
        int steps = (int)((param.sigmaMax - param.sigmaMin) / param.sigmaStep + 1 + 0.00001);
        int k0 = std::lround(param.sigmaMin / param.sigmaStep); // starting point
        for (int i = 0; i < steps; i++) {
            int k = k0 + i;
            double sigma = k * param.sigmaStep;
            param.ChargeRaster.push_back(sigma);
        }
    }

    if (initializeMolecules)
        molecules.clear();

    if (initializeCalculations)
        calculations.clear();
        subcalculations.clear();

    if (showBinarySpecs)
        display("\nBINARY SPECS\n-------------------------\n" + compilation_mode + "\n" + OPENMP_parallelization + "\n" + vectorization_level + "\n" + precision + "\n-------------------------\n\n");
}


void averageAndClusterSegments(parameters& param, molecule& _molecule, int approximateNumberOfSegmentTypes = 0) {

    // save reallocation time by specifying the approximate segment type number
    // this is possible whenever the sigma profile is reloaded and the previous number
    // is known
    if (approximateNumberOfSegmentTypes != 0) {
        // as default a vector resizes to the double of the current size when needed
        // here we are taking 110% of the approximateSegmentTypeNumber hopefully  
        // hindering a reallocation
        approximateNumberOfSegmentTypes = int(1.1 * approximateNumberOfSegmentTypes);
        _molecule.segments.reserve(approximateNumberOfSegmentTypes);
    }

    
    // average the segments
    int numberOfSegments = int(_molecule.segmentAreas.size());
    int numberOfAtoms = int(_molecule.atomAtomicNumbers.size());

    Eigen::VectorXd segmentRadiiSquared = _molecule.segmentAreas / PI;
    double RavSquared = param.Rav * param.Rav;
    double RavCorrSquared = param.RavCorr * param.RavCorr;


    Eigen::VectorXd averagedSigmas = Eigen::VectorXd::Zero(numberOfSegments);
    Eigen::VectorXd averagedSigmaCorrs;

    bool calculateMisfitCorrelation = false;
    if (param.sw_misfit == 1) {
        calculateMisfitCorrelation = true;
        averagedSigmaCorrs = Eigen::VectorXd::Zero(numberOfSegments);
    }
    else if (param.sw_misfit == 2 && (_molecule.moleculeCharge == 0 || numberOfAtoms > 1)) {
        calculateMisfitCorrelation = true;
        averagedSigmaCorrs = Eigen::VectorXd::Zero(numberOfSegments);
    }

    double distanceSegmentSegmentSquared;
    double temporaryValue = 0, multiplyingFactor = 0;

    for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {

        double runningTotalSigmas = 0, runningTotalSigmaCorrs = 0;

        for (int segmentIndexJ = 0; segmentIndexJ < numberOfSegments; segmentIndexJ++) {
            distanceSegmentSegmentSquared = (_molecule.segmentPositions(segmentIndexI, Eigen::indexing::all) - _molecule.segmentPositions(segmentIndexJ, Eigen::indexing::all)).array().square().matrix().sum();

            temporaryValue = segmentRadiiSquared(segmentIndexJ) + RavSquared;
            multiplyingFactor = ((segmentRadiiSquared(segmentIndexJ) * RavSquared) / temporaryValue) * exp(-distanceSegmentSegmentSquared / temporaryValue);

            runningTotalSigmas += multiplyingFactor;
            averagedSigmas(segmentIndexI) = averagedSigmas(segmentIndexI) + _molecule.segmentSigmas(segmentIndexJ) * multiplyingFactor;

            if (calculateMisfitCorrelation == true) {
                temporaryValue = segmentRadiiSquared(segmentIndexJ) + RavCorrSquared;
                multiplyingFactor = ((segmentRadiiSquared(segmentIndexJ) * RavCorrSquared) / temporaryValue) * exp(-distanceSegmentSegmentSquared / temporaryValue);

                runningTotalSigmaCorrs += multiplyingFactor;
                averagedSigmaCorrs(segmentIndexI) = averagedSigmaCorrs(segmentIndexI) + _molecule.segmentSigmas(segmentIndexJ) * multiplyingFactor;
            }
        }
        averagedSigmas(segmentIndexI) = averagedSigmas(segmentIndexI) / runningTotalSigmas;
        
        if (calculateMisfitCorrelation == true)
            averagedSigmaCorrs(segmentIndexI) = averagedSigmaCorrs(segmentIndexI) / runningTotalSigmaCorrs;
    }

    bool calculateSolvationEnergies = param.dGsolv_E_gas.size() > 0;
    if (calculateSolvationEnergies) {
        if (_molecule.qmMethod != "DFT_CPCM_BP86_def2-TZVP+def2-TZVPD_SP" && _molecule.qmMethod != "DFT_BP86_def2-TZVPD_SP") {
            if (param.sw_dGsolv_calculation_strict == 1) {
                throw std::runtime_error("The QSPR model for the molar volume only works for the quantum chemistry method DFT_BP86_def2-TZVPD_SP");
            }
            else {
                warnings.push_back(" - The QSPR model for the molar volume was parametrized using a different quantum chemistry method than the one you are using. Recommended method: DFT_BP86_def2-TZVPD_SP");
            }
        }

        int numberOfSiAtoms = 0;
        int numberOfHAtoms = 0;
        int numberOfOAtoms = 0;
        for (int i = 0; i < numberOfAtoms; i++) {
            if (_molecule.atomAtomicNumbers[i] == 14)
                numberOfSiAtoms += 1;
            else if (_molecule.atomAtomicNumbers[i] == 1 || _molecule.atomAtomicNumbers[i] > 100)
                numberOfHAtoms += 1;
            else if (_molecule.atomAtomicNumbers[i] == 8)
                numberOfOAtoms += 1;
        }

        if (numberOfAtoms == 3 && numberOfHAtoms == 2 && numberOfOAtoms == 1) {
            _molecule.molarVolumeAt25C = 18.06863632;
        }
        else {
            Eigen::ArrayXd averagedSigmasSquared = averagedSigmas.array() * averagedSigmas.array();
            double secondSigmaMoment = (averagedSigmasSquared * _molecule.segmentAreas.array()).sum() * 10000;
            double fourthSigmaMoment = (averagedSigmasSquared * averagedSigmasSquared * _molecule.segmentAreas.array()).sum() * 100000000;

            _molecule.molarVolumeAt25C = 0.9430785419976806 * numberOfAtoms \
                + 0.6977322963011842 * _molecule.Area \
                - 0.3161763939689293 * secondSigmaMoment \
                + 0.032441059832647084 * fourthSigmaMoment \
                + 8.113026329415828 * numberOfSiAtoms \
                - 0.07066832029215675;
        }
    }

    // determine the hydrogen bonding type based on the atomic number

    for (int i = 0; i < numberOfSegments; i++) {

        int atomicNumber = _molecule.segmentAtomicNumber[i];
        // 0 : non HB  | 1 : potential donor  | 2 : potential acceptor | 3 : potential donor or acceptor
        switch (param.HBClassElmnt[atomicNumber]) {
        case 0:
            _molecule.segmentHydrogenBondingType(i) = 0;
            break;
        case 1:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 1 : 0;
            break;
        case 2:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 0 : 2;
            break;
        case 3:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 1 : 2;
            break;
        default:
            throw std::runtime_error("Unknown HB classification value used.");
        }
    }

    // cluster segments into segment types
    double sigmaLeft = -1;
    double sigmaRight = -1;
    double AsigmaLeft = -1;
    double AsigmaRight = -1;

    double sigmaCorrLeft = -1;
    double sigmaCorrRight = -1;

    double AsigmaLeftSigmaCorrLeft = -1;
    double AsigmaLeftSigmaCorrRight = -1;
    double AsigmaRightSigmaCorrLeft = -1;
    double AsigmaRightSigmaCorrRight = -1;

    unsigned short ind_SigmaCorr_left = 0;

    for (int j = 0; j < numberOfSegments; j++) {

        //unsigned short ind_Sigma_left = int((averagedSigmas(j) - param.sigmaMin) / param.sigmaStep);
        double x = (averagedSigmas(j) - param.sigmaMin) / param.sigmaStep;
        unsigned short ind_Sigma_left = static_cast<unsigned short>(std::floor(x + 1e-12));

        sigmaLeft = param.ChargeRaster[ind_Sigma_left];
        sigmaRight = param.ChargeRaster[ind_Sigma_left + 1];

        AsigmaRight = _molecule.segmentAreas(j) * (averagedSigmas(j) - sigmaLeft) / param.sigmaStep;
        AsigmaLeft = _molecule.segmentAreas(j) * (sigmaRight - averagedSigmas(j)) / param.sigmaStep;
        if (AsigmaLeft < 0 || AsigmaRight < 0) {
            throw std::runtime_error("Calculated negative segment area. This should not happen.");
        }

        if (calculateMisfitCorrelation == true) {
            ind_SigmaCorr_left = int((averagedSigmaCorrs(j) - param.sigmaMin) / param.sigmaStep);

            sigmaCorrLeft = param.ChargeRaster[ind_SigmaCorr_left];
            sigmaCorrRight = param.ChargeRaster[ind_SigmaCorr_left + 1];

            AsigmaLeftSigmaCorrRight = AsigmaLeft * (averagedSigmaCorrs(j) - sigmaCorrLeft) / param.sigmaStep;
            AsigmaLeftSigmaCorrLeft = AsigmaLeft * (sigmaCorrRight - averagedSigmaCorrs(j)) / param.sigmaStep;
            AsigmaRightSigmaCorrRight = AsigmaRight * (averagedSigmaCorrs(j) - sigmaCorrLeft) / param.sigmaStep;
            AsigmaRightSigmaCorrLeft = AsigmaRight * (sigmaCorrRight - averagedSigmaCorrs(j)) / param.sigmaStep;
        }


        unsigned short atomicNumber = _molecule.segmentAtomicNumber(j);
        if (param.sw_atomicNumber == 0) {
            atomicNumber = 0;
        }

        // for monoatomic ions or if correlation is deactivated
        if (_molecule.moleculeGroup == 3 || _molecule.moleculeGroup == 5 || calculateMisfitCorrelation == false) {

            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, 0.0f, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeft, _molecule.index);
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, 0.0f, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRight, _molecule.index);
        }
        else {
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, sigmaCorrLeft, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeftSigmaCorrLeft, _molecule.index);
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, sigmaCorrRight, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeftSigmaCorrRight, _molecule.index);
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, sigmaCorrLeft, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRightSigmaCorrLeft, _molecule.index);
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, sigmaCorrRight, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRightSigmaCorrRight, _molecule.index);
        }
    }
}


molecule loadNewMolecule(parameters& param, std::string componentPath) {

    molecule newMolecule;
    if (param.sw_COSMOfiles_type == "Turbomole_COSMO_TZVP" || param.sw_COSMOfiles_type == "Turbomole_COSMO_TZVPD_FINE") {
        newMolecule = getMoleculeFromTurbomoleCOSMOfile(componentPath);
    }
    else if (param.sw_COSMOfiles_type == "ORCA_COSMO_TZVPD") {
        newMolecule = getMoleculeFromORCACOSMOfile(componentPath);
    }
    else if (param.sw_COSMOfiles_type == "ML_ORCA_COSMO_TZVPD_SP") {
        newMolecule = getMoleculeFromJSONfile(componentPath);
    }
    else {
        throw std::runtime_error("No method for reading COSMOfiles has been implemented for the following type: " + param.sw_COSMOfiles_type);
    }

    std::string componentName = componentPath;
    std::vector<std::string> parts = split(componentName, '\\');
    parts = split(parts[parts.size() - 1], '/');
    parts = split(parts[parts.size() - 1], '.');
    newMolecule.name = trim(parts[0]);

    // if sigma profile not yet loaded
    if (newMolecule.segments.size() == 0) {
        double sumOfScreeningCharge = (newMolecule.segmentAreas.array() * newMolecule.segmentSigmas.array()).matrix().sum();
        newMolecule.moleculeCharge = (signed char)(std::round(-1.0f * sumOfScreeningCharge));

        int numberOfAtoms = int(newMolecule.atomAtomicNumbers.size());

        // Store atomic radii and check for consistency
        for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
            int AtomicNumber = newMolecule.atomAtomicNumbers(atomIndex);
            if (param.R_i_COSMO[AtomicNumber] != 0 && newMolecule.atomRadii(atomIndex) != param.R_i_COSMO[AtomicNumber]) {
                throw std::runtime_error("Inconsistent radii set for atomic number " + std::to_string(AtomicNumber) + " was found.");
            }
            else if (param.R_i_COSMO[AtomicNumber] == 0 && newMolecule.atomRadii(atomIndex) != 0) {
                param.R_i_COSMO[AtomicNumber] = newMolecule.atomRadii(atomIndex);
            }
        }

        // Calculate distance between atoms
        Eigen::MatrixXd distanceAtomAtomSquared(numberOfAtoms, numberOfAtoms);

        for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
            Eigen::MatrixXd thisAtomPosition = newMolecule.atomPositions(atomIndex, Eigen::indexing::all);
            distanceAtomAtomSquared(Eigen::indexing::all, atomIndex) = (newMolecule.atomPositions - thisAtomPosition.replicate(numberOfAtoms, 1)).array().square().rowwise().sum();
        }

        // set moleculeGroup
        int moleculeGroup = 0;

        if (param.sw_differentiateMoleculeGroups == 1) {

            moleculeGroup = -1;

            if (numberOfAtoms == 3) {
                int numberOfFoundOxygens = 0;
                int numberOfFoundHydrogens = 0;
                for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
                    if (newMolecule.atomAtomicNumbers(atomIndex) == 1) {
                        numberOfFoundHydrogens += 1;
                    }
                    else if (newMolecule.atomAtomicNumbers(atomIndex) == 8) {
                        numberOfFoundOxygens += 1;
                    }
                }

                if (numberOfFoundOxygens == 1 && numberOfFoundHydrogens == 2) {
                    moleculeGroup = 2;
                }
            }

            if (moleculeGroup == -1) {
                if (newMolecule.moleculeCharge == 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 0 : 1;
                }
                else if (newMolecule.moleculeCharge > 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 3 : 4;
                }
                else if (newMolecule.moleculeCharge < 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 5 : 6;
                }
            }
        }
        newMolecule.moleculeGroup = (unsigned short)moleculeGroup;

        // this changes the atomic number of a hydrogen atom to 100 + the atomic number of the closest heavy atom
        // giving the abbility to differentiate between hydrogen atoms depending on the atom they are bound to.
        if (param.sw_differentiateHydrogens == 1) {

            for (int atomIndexI = 0; atomIndexI < numberOfAtoms; atomIndexI++) {
                double minimumDistanceAtomIAtomJSquared = 10e14;
                int closestAtomIndex = -1;
                if (newMolecule.atomAtomicNumbers(atomIndexI) != 1) {
                    continue;
                }

                for (int atomIndexJ = 0; atomIndexJ < numberOfAtoms; atomIndexJ++) {

                    if (atomIndexI == atomIndexJ) {
                        continue;
                    }

                    // search for next atom that is not a Hydrogen unless tmolecule is H2
                    if (newMolecule.atomAtomicNumbers(atomIndexJ) == 1 && newMolecule.atomAtomicNumbers(atomIndexJ) < 100 && numberOfAtoms != 2) {
                        continue;
                    }

                    if (distanceAtomAtomSquared(atomIndexI, atomIndexJ) < minimumDistanceAtomIAtomJSquared) {
                        minimumDistanceAtomIAtomJSquared = distanceAtomAtomSquared(atomIndexI, atomIndexJ);
                        closestAtomIndex = atomIndexJ;
                    }
                }
                int newAtomicNumber = 100 + newMolecule.atomAtomicNumbers(closestAtomIndex);
                newMolecule.atomAtomicNumbers(atomIndexI) = newAtomicNumber;
                param.R_i_COSMO[newAtomicNumber] = param.R_i_COSMO[1];
            }
        }
        else if (param.sw_differentiateHydrogens != 0) {
            throw std::runtime_error("differentiateHydrogens accepts values [0, 1]");
        }

        int numberOfSegments = int(newMolecule.segmentAreas.size());
        newMolecule.segmentAtomicNumber = Eigen::VectorXi(numberOfSegments);

        for (int i = 0; i < numberOfSegments; i++) {

            int atomicNumber = newMolecule.atomAtomicNumbers(newMolecule.segmentAtomIndices(i));
            newMolecule.segmentAtomicNumber(i) = atomicNumber;
        }

        newMolecule.segmentHydrogenBondingType = Eigen::VectorXi(numberOfSegments);
        averageAndClusterSegments(param, newMolecule);
    }

    newMolecule.clear_unneeded_matrices(param.sw_alwaysReloadSigmaProfiles);
    newMolecule.segments.shrink_to_fit();

#ifdef PRINT_DEBUG_INFO
    //newMolecule.segments.sort();
    //WriteExtendedSigmaProfileToFile(newMolecule.name + ".extsp", newMolecule.segments);
#endif

    return newMolecule;
}


molecule createHoleMolecule(parameters& param) {
    molecule holeMolecule;

    holeMolecule.name = "hole";
    holeMolecule.index = int(molecules.size());
    holeMolecule.moleculeCharge = 0;/////
    holeMolecule.Area = param.hole_area;
    holeMolecule.Volume = 0; // this parameter is only to fullfill the requirements for a standard molecule struct
    // it is never used in calculations as the volume of the holes is dependent on concentrations

    holeMolecule.moleculeGroup = 7;
    holeMolecule.atomPositions = Eigen::MatrixXd::Zero(1, 3);
    holeMolecule.atomAtomicNumbers = Eigen::VectorXi::Zero(1);
    holeMolecule.atomRadii = Eigen::VectorXd::Zero(1);

    holeMolecule.segmentAreas = Eigen::VectorXd::Constant(1, holeMolecule.Area);/////
    holeMolecule.segmentPositions = Eigen::MatrixXd::Zero(1, 3);/////
    holeMolecule.segmentAtomicNumber = Eigen::VectorXi::Zero(1);/////
    holeMolecule.segmentAtomIndices = Eigen::VectorXi::Zero(1); // 299?
    holeMolecule.segmentHydrogenBondingType = Eigen::VectorXi::Zero(1);
    holeMolecule.segmentSigmas = Eigen::VectorXd::Zero(1);/////

    averageAndClusterSegments(param, holeMolecule);

    holeMolecule.clear_unneeded_matrices(param.sw_alwaysReloadSigmaProfiles);
    holeMolecule.segments.shrink_to_fit();

    return holeMolecule;
}


void reloadAllMolecules() {
    threadException e;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < molecules.size(); i++) {
        e.run([=] {
            molecule _molecule = *molecules[i];
            int previousNumberOfSegmentTypes = int(_molecule.segments.size());
            _molecule.segments.clear();
            averageAndClusterSegments(param, _molecule, previousNumberOfSegmentTypes);
            });
    }
    e.rethrow();
}


void resizeMonoatomicCations(parameters& param, std::vector<std::shared_ptr<molecule>> molecules) {
#ifdef MEASURE_TIME
    std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
    // scale A and V for monoatomic cations
    for (int i = 0; i < molecules.size(); i++) {

        if (molecules[i]->moleculeGroup == 3) {
            int AN = molecules[i]->atomAtomicNumbers(0);
            double R_i = param.R_i[AN];
            molecules[i]->Area = (4 * PI * R_i * R_i);
            molecules[i]->Volume = (4.0 / 3 * PI * R_i * R_i * R_i);
        }
    }
#ifdef MEASURE_TIME
    rescaleSegments_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count());
#endif
}


void rescaleSegments(parameters& param, calculation& _calculation) {
    // rescale segments for monoatomic cations only if present
    if (_calculation.segments.numberOfSegmentsForGroup[3] != 0) {

        // monoatomic cation
        std::unordered_map<unsigned short, int[2]> segmentIndicesBelongingToASpecificAtomicNumber;
        for (int j = _calculation.segments.lowerBoundIndexForGroup[3];
            j < _calculation.segments.upperBoundIndexForGroup[3]; j++) {

            unsigned short AN = _calculation.segments.SegmentTypeAtomicNumber[j];

            if (segmentIndicesBelongingToASpecificAtomicNumber.find(AN) == segmentIndicesBelongingToASpecificAtomicNumber.end()) {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][0] = (unsigned short)j;
            }
            else {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][1] = (unsigned short)j;
            }

        }
        for (auto it = segmentIndicesBelongingToASpecificAtomicNumber.begin(); it != segmentIndicesBelongingToASpecificAtomicNumber.end(); ++it) {

            int AN = it->first;
            int ind_left = it->second[0];
            int ind_right = it->second[1];

            if (ind_right - ind_left > 1) {
                throw std::runtime_error("More than 2 segmentTypes was found for monoatomic cation: " + std::to_string(AN) + ".");
            }

            int ind_molecule = -1;

            for (int k = 0; _calculation.segments.SegmentTypeAreas[ind_left].size(); k++) {
                if (_calculation.segments.SegmentTypeAreas[ind_left][k] > 0.0) {
                    ind_molecule = k;
                    break;
                }
            }

            int screeningCharge = int(_calculation.components[ind_molecule]->moleculeCharge * -1);

            double newArea = (4 * PI * param.R_i[AN] * param.R_i[AN]);
            double newSigma = screeningCharge / newArea;

            //unsigned short ind_Sigma_left = int((newSigma - param.sigmaMin) / param.sigmaStep);
            double x = (newSigma - param.sigmaMin) / param.sigmaStep;
            unsigned short ind_Sigma_left = static_cast<unsigned short>(std::floor(x + 1e-12));

            double sigmaLeft = param.ChargeRaster[ind_Sigma_left];
            double sigmaRight = param.ChargeRaster[ind_Sigma_left + 1];

            double AsigmaRight = newArea * (newSigma - sigmaLeft) / param.sigmaStep;
            double AsigmaLeft = newArea * (sigmaRight - newSigma) / param.sigmaStep;

            _calculation.segments.SegmentTypeAreas[ind_left][ind_molecule] = AsigmaLeft;
            _calculation.segments.SegmentTypeSigma[ind_left] = sigmaLeft;

            _calculation.segments.SegmentTypeAreas[ind_right][ind_molecule] = AsigmaRight;
            _calculation.segments.SegmentTypeSigma[ind_right] = sigmaRight;

        }
    }
}


void calculateSegmentConcentrations(calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate = {}) {

    if (actualConcentrationIndicesToCalculate.size() == 0) {
        actualConcentrationIndicesToCalculate = std::vector<int>(_calculation.concentrations.size());
        std::iota(std::begin(actualConcentrationIndicesToCalculate), std::end(actualConcentrationIndicesToCalculate), 0);
    }

    //if (_calculation.components[0]->name == "H2O_c000") {
    //    for (int k = 0; k < _calculation.segments.SegmentTypeAreas.size(); k++) {
    //        _calculation.segments.SegmentTypeAreas[k][0] = param.exp_param.at("scaling_factor") * _calculation.segments.SegmentTypeAreas[k][0];
    //    }
    //}

    // calculate the mole fraction of segments for each concentration
    for (auto j : actualConcentrationIndicesToCalculate) {

        int firstNonZeroSegmentIndex = 0;
        int lastNonZeroSegmentIndex = int(_calculation.concentrations.size()) - 1;

        std::vector<double> segmentConcentration(_calculation.segments.size(), 0.0);

        double sumAreaSegmentsConcentrationj = 0.0;

        for (int k = 0; k < _calculation.segments.size(); k++) {

            double areaSegmentK = 0.0;

            for (int m = 0; m < _calculation.components.size(); m++) {
                double thisArea = _calculation.concentrations[j][m] * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK += thisArea;
                sumAreaSegmentsConcentrationj += thisArea;
            }

            segmentConcentration[k] = areaSegmentK;
        }

        double cumulativeSumOfAreasFromSegmentZeroOn = 0.0;

        bool firstNonZeroSegmentIndex_found = false;
        for (int k = 0; k < _calculation.segments.size(); k++) {

            cumulativeSumOfAreasFromSegmentZeroOn += segmentConcentration[k];

            if (cumulativeSumOfAreasFromSegmentZeroOn != 0 && !firstNonZeroSegmentIndex_found) {
                firstNonZeroSegmentIndex = k;
                firstNonZeroSegmentIndex_found = true;
            }
            if (segmentConcentration[k] != 0) {
                lastNonZeroSegmentIndex = k;
            }

            _calculation.segmentConcentrations(k, j) = calcType(segmentConcentration[k] / sumAreaSegmentsConcentrationj);
        }

        _calculation.lowerBoundIndexForCOSMOSPACECalculation[j] = RoundDownToNextMultipleOfEight(firstNonZeroSegmentIndex);
        _calculation.upperBoundIndexForCOSMOSPACECalculation[j] = RoundUpToNextMultipleOfEight(lastNonZeroSegmentIndex + 1);
    }
}


void calculatePartialSegmentConcentrationsPartialv(calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate, double PartialholeConcentrationPartialv) {

    // calculate the mole fraction of segments for each concentration
    for (auto j : actualConcentrationIndicesToCalculate) {

        double holeIndex = int(_calculation.components.size() - 1);
        double x_h = _calculation.concentrations[j][holeIndex];

        int firstNonZeroSegmentIndex = 0;
        int lastNonZeroSegmentIndex = int(_calculation.concentrations.size()) - 1;

        std::vector<double> segmentConcentration(_calculation.segments.size(), 0.0);
        std::vector<double> PartialsegmentConcentrationPartialv_2(_calculation.segments.size(), 0.0);

        double sumAreaSegmentsConcentrationj = 0.0;
        double PartialsumAreaSegmentsConcentrationjPartialv_2 = 0.0;
        double sumAreaSegmentsHole = 0.0;

        for (int k = 0; k < _calculation.segments.size(); k++) {

            double areaSegmentK = 0.0;
            double thisArea = 0.0;
            double areaSegmentK_2 = 0.0;
            double thisArea_2 = 0.0;

            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                thisArea = _calculation.concentrations[j][m] * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK += thisArea;
                sumAreaSegmentsConcentrationj += thisArea;
                thisArea_2 = _calculation.concentrations[j][m] / (1 - x_h) * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK_2 += thisArea_2;
                PartialsumAreaSegmentsConcentrationjPartialv_2 += thisArea_2;
            }
            thisArea = _calculation.concentrations[j][holeIndex] * _calculation.segments.SegmentTypeAreas[k][holeIndex];
            thisArea_2 = -1 * _calculation.segments.SegmentTypeAreas[k][holeIndex];
            areaSegmentK += thisArea;
            areaSegmentK_2 += thisArea_2;
            PartialsumAreaSegmentsConcentrationjPartialv_2 += thisArea_2;
            sumAreaSegmentsConcentrationj += thisArea;

            sumAreaSegmentsHole += _calculation.segments.SegmentTypeAreas[k][holeIndex];

            segmentConcentration[k] = areaSegmentK;
            PartialsegmentConcentrationPartialv_2[k] = areaSegmentK_2;
        }

        for (int k = 0; k < _calculation.segments.size(); k++) {

            //double nominator = (-PartialsegmentConcentrationPartialv[k] + _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) * ((1 - x_h) * PartialsumAreaSegmentsConcentrationjPartialv + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) - ((1 - x_h) * PartialsegmentConcentrationPartialv[k] + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) * (-PartialsumAreaSegmentsConcentrationjPartialv + _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]);
            //double denominator = std::pow((1 - x_h) * PartialsumAreaSegmentsConcentrationjPartialv + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1], 2);

            double nominator_derivative = PartialsegmentConcentrationPartialv_2[k] * (-1) * PartialholeConcentrationPartialv;
            double nominator = segmentConcentration[k];
            double denominator_derivative = PartialsumAreaSegmentsConcentrationjPartialv_2 * (-1) * PartialholeConcentrationPartialv;
            double denominator = sumAreaSegmentsConcentrationj;

            _calculation.PartialsegmentConcentrationsPartialv(k, j) = calcType((nominator_derivative * denominator - nominator * denominator_derivative)/(std::pow(denominator, 2)));
            //_calculation.PartialsegmentConcentrationsPartialv(k, j) = calcType(PartialholeConcentrationPartialv * (_calculation.segments.SegmentTypeAreas[k][holeIndex] * denominator - nominator * sumAreaSegmentsHole) / (std::pow(denominator, 2)));
        }
    }
}


void finishCalculationInitiation(calculation& _calculation, parameters& param) {

    if (_calculation.concentrations.size() > 65535) {
        throw std::runtime_error("Too many calculations, other datatype would be necessary for newCalculation.referenceStates to cope with this amount. (unsigned short used so far allowing for up to 65535)");
    }

    // check charge of all concentrations
    for (int j = 0; j < _calculation.concentrations.size(); j++) {

        double mix_chrg = 0;
        for (int k = 0; k < _calculation.components.size(); k++) {
            mix_chrg = mix_chrg + _calculation.components[k]->moleculeCharge * _calculation.concentrations[j][k];
        }

        if (abs(mix_chrg) > MAX_CONCENTRATION_DIFF_FROM_ZERO) {
            throw std::runtime_error("For calculation number " + std::to_string(_calculation.number) + " the mixture number " + std::to_string(j) + " is not electroneutral. residual charge: " + std::to_string(abs(mix_chrg)));
        }
    }

    // save sorting of concentrations according to conditions, this clusters the calculation of the interaction matrix
    // sorting by similar concentration does not make sense as the gammas from the previous iteration
    // are saved for every concentration already accelerating the COSMOSPACE convergence
    std::vector<int> sortingVector(_calculation.concentrations.size());
    std::iota(sortingVector.begin(), sortingVector.end(), 0);
    std::sort(sortingVector.begin(), sortingVector.end(),
        [&](int i, int j) {
            if (_calculation.temperatures[i] != _calculation.temperatures[j])
                return _calculation.temperatures[i] < _calculation.temperatures[j];
            else
                return i < j;
        });

    apply_vector_permutation_in_place(_calculation.concentrations, sortingVector);
    apply_vector_permutation_in_place(_calculation.temperatures, sortingVector);

    _calculation.actualConcentrationIndices = std::vector<int>(_calculation.concentrations.size());
    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        _calculation.actualConcentrationIndices[sortingVector[j]] = j;
    }

    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        int TauIndex = _calculation.addOrFindTauIndexForConditions(_calculation.temperatures[j]);
        _calculation.TauConcentrationIndices[TauIndex].push_back(j);
    }

    // initiate arrays for the calculation
    _calculation.segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasSecondPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()));
    _calculation.PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()));

    _calculation.PhiDash_pxi = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    _calculation.ThetaDash_pxi = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());

    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        _calculation.lowerBoundIndexForCOSMOSPACECalculation.push_back(0);
        _calculation.upperBoundIndexForCOSMOSPACECalculation.push_back(int(_calculation.segments.size()));
    }

    _calculation.shrink_to_fit();
}


calculation loadSubcalculation(std::vector<std::shared_ptr<molecule>> molecules, std::vector<double> temperatures, std::vector<std::vector<double>> concentrationsToLoad, std::vector<double> pressures = { 0.0 }) {

    size_t numberOfComponentsToLoadSegmentsOf = molecules.size();
    size_t numberOfComponents = numberOfComponentsToLoadSegmentsOf - 1;

    calculation newCalculation = calculation(numberOfComponentsToLoadSegmentsOf);

    for (int j = 0; j < numberOfComponentsToLoadSegmentsOf; j++) {
        int moleculeIndex = j;

        std::shared_ptr<molecule> thisMolecule = molecules[moleculeIndex];

        for (int k = 0; k < thisMolecule->segments.size(); k++) {
            newCalculation.segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
                thisMolecule->segments.SegmentTypeSigma[k],
                thisMolecule->segments.SegmentTypeSigmaCorr[k],
                thisMolecule->segments.SegmentTypeHBtype[k],
                thisMolecule->segments.SegmentTypeAtomicNumber[k],
                thisMolecule->segments.SegmentTypeAreas[k][0],
                thisMolecule->index);
        }

        newCalculation.components.push_back(thisMolecule);
        newCalculation.moleculeIndices[j] = moleculeIndex;
    }

    newCalculation.segments.sort();
    newCalculation.segments.shrink_to_fit();

    newCalculation.temperatures = temperatures;
    newCalculation.temperatures.insert(newCalculation.temperatures.end(), temperatures.begin(), temperatures.end());

    size_t numberOfConcentrations = temperatures.size();
    Eigen::MatrixXd concentrations(numberOfConcentrations * 2, numberOfComponents);
    for (int k = 0; k < numberOfConcentrations; k++) {
        for (int l = 0; l < numberOfComponents; l++) {
            concentrations(k, l) = concentrationsToLoad[k][l];
            concentrations(k + numberOfConcentrations, l) = concentrationsToLoad[k][l];
        }
    }

    // concentrations and temperatures

    for (int j = 0; j < (size_t)numberOfConcentrations * 2; j++) {

        std::vector<double> rowConcentration;

        for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {
            double val = 0.0f;
            if (k < numberOfComponents)
                val = (double)concentrations(j, k); // for pure component here was a 0 instead of k

            rowConcentration.push_back(val);
        }

        newCalculation.concentrations.push_back(rowConcentration);
    }

    Eigen::MatrixXd referenceStateConcentrations = Eigen::MatrixXd::Zero(numberOfConcentrations * 2, numberOfComponents);


    newCalculation.originalNumberOfCalculations = (unsigned short)newCalculation.concentrations.size();

    // reference states
    Eigen::MatrixXi referenceStateTypes(numberOfConcentrations * 2, numberOfComponents);
    for (int k = 0; k < numberOfConcentrations * 2; k++) {
        for (int l = 0; l < numberOfComponents; l++) {
            referenceStateTypes(k, l) = (int)5;
        }
    }
    for (int j = 0; j < numberOfConcentrations * 2; j++) {

        int referenceStateType = referenceStateTypes(j, 0);

        newCalculation.referenceStateType.push_back((unsigned short)referenceStateType);

        std::vector<int> thisReferenceStateCalculationIndices;
        for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {

            std::vector<double> referenceStateConcentration(numberOfComponentsToLoadSegmentsOf, 0.0f);
            referenceStateConcentration[numberOfComponentsToLoadSegmentsOf - 1] = 1.0f;

            double temperature = (double)newCalculation.temperatures[j];
            int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);
            thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);
        }
        newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
    }

    newCalculation.concentrations_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
    newCalculation.concentrations_data.setZero();
    for (int j = 0; j < numberOfConcentrations; j++) {
        for (int k = 0; k < numberOfComponents; k++) {
            newCalculation.concentrations_data(j, k) = concentrationsToLoad[j][k];
            newCalculation.concentrations_data(numberOfConcentrations + j, k) = concentrationsToLoad[j][k];
        }
    }

    newCalculation.originalConcentrations.~Map();
    new (&newCalculation.originalConcentrations) Eigen::Map<Eigen::MatrixXd>(
        newCalculation.concentrations_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponents);

    for (int j = 0; j < numberOfConcentrations; j++) {
        for (int k = 0; k < numberOfComponents; k++) {
            newCalculation.originalConcentrations(j, k) = concentrationsToLoad[j][k];
            newCalculation.originalConcentrations(numberOfConcentrations + j, k) = concentrationsToLoad[j][k];
        }
    }

    //newCalculation.originalConcentrations = newCalculation.concentrations_data;

    newCalculation.lnGammaCombinatorial_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaCombinatorial_data.setZero();
    new (&newCalculation.lnGammaCombinatorial) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaCombinatorial_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.lnGammaResidual_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaResidual_data.setZero();
    new (&newCalculation.lnGammaResidual) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaResidual_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.lnGammaTotal_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaTotal_data.setZero();
    new (&newCalculation.lnGammaTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaTotal_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.number = (int)0;
    finishCalculationInitiation(newCalculation, param);

    if (param.sw_phi == 1) {

        newCalculation.molarVolume_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.molarVolume_data.setZero();
        new (&newCalculation.molarVolume) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.molarVolume_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.repulsivePressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.repulsivePressure_data.setZero();
        new (&newCalculation.repulsivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.repulsivePressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.attractivePressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.attractivePressure_data.setZero();
        new (&newCalculation.attractivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.attractivePressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.totalPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.totalPressure_data(j, 0) = double(pressures[0]);
        }
        new (&newCalculation.totalPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.totalPressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.targetPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.targetPressure_data(j, 0) = double(pressures[0]);
        }
        new (&newCalculation.targetPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.targetPressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.targetTemperature_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.targetTemperature_data(j, 0) = double(temperatures[0]);
        }
        new (&newCalculation.targetTemperature) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.targetTemperature_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.lnPhiTotal_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiTotal_data.setZero();
        new (&newCalculation.lnPhiTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiTotal_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.lnPhiRepulsive_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiRepulsive_data.setZero();
        new (&newCalculation.lnPhiRepulsive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiRepulsive_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.lnPhiAttractive_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiAttractive_data.setZero();
        new (&newCalculation.lnPhiAttractive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiAttractive_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.criticalP_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.criticalP_data.setZero();
        new (&newCalculation.criticalP) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.criticalP_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.criticalT_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.criticalT_data.setZero();
        new (&newCalculation.criticalT) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.criticalT_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.criticalx_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.criticalx_data.setZero();
        new (&newCalculation.criticalx) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.criticalx_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        //newCalculation.molarResidualEnthalpy = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
        //newCalculation.molarResidualHeatCapacityAtConstantPressure = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
        
        newCalculation.molarResidualEnthalpy_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.molarResidualEnthalpy_data.setZero();
        new (&newCalculation.molarResidualEnthalpy) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.molarResidualEnthalpy_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.molarResidualHeatCapacityAtConstantPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.molarResidualHeatCapacityAtConstantPressure_data.setZero();
        new (&newCalculation.molarResidualHeatCapacityAtConstantPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.molarResidualHeatCapacityAtConstantPressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));
        
        newCalculation.lnPhiTotalPDETAtConstantPx = MatrixCalcType::Zero(newCalculation.originalNumberOfCalculations, newCalculation.components.size());
        newCalculation.lnGammaResidualPDETatConstantvx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
        newCalculation.lnGammaResidualSecondPDETatConstantvx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
        newCalculation.lnGammaResidualPDEvatConstantTx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);

        newCalculation.recalculateConcentrationsForPhi = true;
    }

    return newCalculation;
}


void calculateLnGammaCombinatorial(parameters& param, calculation& _calculation) {

    std::vector<double> averageVolumes(_calculation.concentrations.size(), 0.0);
    std::vector<double> averageAreas(_calculation.concentrations.size(), 0.0);

    // Volume fraction to mole fraction ratio for all compositions
    // Area fraction to mole fraction ratio for all compositions
    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        for (int k = 0; k < _calculation.components.size(); k++) {
            averageVolumes[j] += _calculation.components[k]->Volume * _calculation.concentrations[j][k];
            averageAreas[j] += _calculation.components[k]->Area * _calculation.concentrations[j][k];
        }

        for (int k = 0; k < _calculation.components.size(); k++) {
            _calculation.PhiDash_pxi(j, k) = _calculation.components[k]->Volume / averageVolumes[j];
            _calculation.ThetaDash_pxi(j, k) = _calculation.components[k]->Area / averageAreas[j];
        }
    }

    std::vector<double> qi_std(_calculation.components.size());

    /* Calculation of molecule specific parameters for combinatorial contribution */
    for (int i = 0; i < _calculation.components.size(); i++) {
        qi_std[i] = _calculation.components[i]->Area / param.comb_SG_A_std;
    }

    Eigen::MatrixXd lnGamaForCalculations = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());

    if (param.sw_combTerm == 1) { // tested
        /* Staverman-Guggenheim term
           cp. Kikic (1980) or Lin & Sandler (2002)
           Remark: square brackets in L&S paper for the li term are wrong as can be seen in monograph Prausnitz Lichtenthaler */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 = _calculation.PhiDash_pxi(i, j) / _calculation.ThetaDash_pxi(i, j);
                lnGamaForCalculations(i, j) = log(_calculation.PhiDash_pxi(i, j)) + 1 - _calculation.PhiDash_pxi(i, j) - \
                    param.comb_SG_z_coord * 0.5 * qi_std[j] * (log(buffdb1) + 1 - buffdb1);
            }
        }
    }
    else if (param.sw_combTerm == 2) { // tested
        /* Klamt (2003) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {
            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 = buffdb1 + _calculation.concentrations[i][j] * log(_calculation.components[j]->Volume);
            }

            for (int j = 0; j < _calculation.components.size(); j++) {
                lnGamaForCalculations(i, j) = (param.comb_lambda0 * (log(_calculation.components[j]->Volume) - buffdb1)) \
                    - (param.comb_lambda1 * (_calculation.PhiDash_pxi(i, j) - 1)) \
                    - (param.comb_lambda2 * (_calculation.ThetaDash_pxi(i, j) - 1));
            }
        }
    }
    else if (param.sw_combTerm == 3) { // tested
        /*  mod. Staverman-Guggenheim with exponential scaling, compare e.g. Soares (2011), Kikic (1980), Donohue Prausnitz (1975) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 += pow(_calculation.components[j]->Volume, param.comb_modSG_exp) * _calculation.concentrations[i][j];
            }

            double buffdb2 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb2 = _calculation.PhiDash_pxi(i, j) / _calculation.ThetaDash_pxi(i, j);

                double PhiDash_pxi_mod_i = pow(_calculation.components[j]->Volume, param.comb_modSG_exp) / buffdb1;

                lnGamaForCalculations(i, j) = log(PhiDash_pxi_mod_i) + 1 - PhiDash_pxi_mod_i - \
                    param.comb_SG_z_coord * 0.5 * qi_std[j] * (log(buffdb2) + 1 - buffdb2);
            }
        }
    }
    else if (param.sw_combTerm == 4) { // not tested yet
        /*  mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double sum_qi_div_xi = 0;
            double sum_ri_div_xi = 0;
            double sum_qi_times_xi = 0;
            double sum_ri_times_xi = 0;

            for (int j = 0; j < _calculation.components.size(); j++) {
                sum_qi_div_xi = sum_qi_div_xi + (_calculation.components[j]->Area / _calculation.concentrations[i][j]);
                sum_ri_div_xi = sum_ri_div_xi + (_calculation.components[j]->Volume / _calculation.concentrations[i][j]);
                sum_qi_times_xi = sum_qi_times_xi + (_calculation.components[j]->Area * _calculation.concentrations[i][j]);
                sum_ri_times_xi = sum_ri_times_xi + (_calculation.components[j]->Volume * _calculation.concentrations[i][j]);
            }

            std::vector<double> lnGamaForThisCalculation;
            for (int j = 0; j < _calculation.components.size(); j++) {

                double qi_hash = (1 - _calculation.concentrations[i][j]) * sum_qi_div_xi;
                double ri_hash = (1 - _calculation.concentrations[i][j]) * sum_ri_div_xi;

                double qi_min = std::min(_calculation.components[j]->Area, qi_hash);
                double qi_max = std::max(_calculation.components[j]->Area, qi_hash);

                double ri_min = std::min(_calculation.components[j]->Volume, ri_hash);
                double ri_max = std::max(_calculation.components[j]->Volume, ri_hash);

                double Fi = pow(_calculation.components[j]->Area / sum_qi_times_xi, param.comb_SGG_lambda * (1 - (qi_min / qi_max)));
                double Vi = pow(_calculation.components[j]->Volume / sum_ri_times_xi, param.comb_SGG_beta * (1 - (ri_min / ri_max)));

                lnGamaForCalculations(i, j) = 1 - Vi - log(Vi) + 1 - (Vi / Fi) - log(Vi / Fi);
            }
        }
    }
    else if (param.sw_combTerm == 5) { // not tested yet
        /*  Franke & Hannebauer (2011) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {
            for (int j = 0; j < _calculation.components.size(); j++) {
                lnGamaForCalculations(i, j) = param.comb_lambda0 * log(_calculation.components[j]->Volume) \
                    + param.comb_lambda1 * (1 - (_calculation.components[j]->Volume / averageVolumes[i]) - log(averageVolumes[i])) \
                    + param.comb_lambda2 * (1 - (_calculation.components[j]->Area / averageAreas[i]) - log(averageAreas[i]));
            }
        }
    }
    else if (param.sw_combTerm != 0) {
        throw std::runtime_error("Error: Invalid switch value for combinatorial term.\n");
    }


    /* Convert activity coefficients to correct reference states. */
    for (int h = 0; h < _calculation.originalNumberOfCalculations; h++) {

        int i = _calculation.actualConcentrationIndices[h];

        for (int j = 0; j < _calculation.components.size(); j++) {

            double multiplier = 1.0;
            // for an ion with reference state PureComponentsOnlyNeutral
            if (_calculation.referenceStateType[h] == 1 && _calculation.referenceStateCalculationIndices[h][j] == -1) {
                multiplier = NAN;
            }
            _calculation.lnGammaCombinatorial(h, j) = multiplier * lnGamaForCalculations(i, j);
#ifdef PRINT_DEBUG_INFO
            display("lnGammaCombinatorial_1_" + std::to_string(h) + ": " + std::to_string(_calculation.lnGammaCombinatorial(h, j)) + "\n");
#endif

            // only subtract reference state value if reference state type is not COSMO
            if (_calculation.referenceStateType[h] != 3 && _calculation.referenceStateType[h] != 4) {
                int indexOfReferenceStateCalculation = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[h][j]];

                _calculation.lnGammaCombinatorial(h, j) -= lnGamaForCalculations(indexOfReferenceStateCalculation, j);
            }
        }
    }
}


// the COSMOSPACE equations are solved by successive substitution
// instead of calculating the complete matrix-vector product and then calculating the gammas for the next iteration
// each row is calculated and the gamma for the row just calculated is already used in the calculation of the next row.
// This was done firstly by accident, but it was found, that it accelerated the convergence by a factor of at least 4.
void calculateLnGammaResidual(parameters& param, calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate, bool withTDerivative = false, bool withvDerivative = false) {

    const int numberOfSegments = int(_calculation.segments.size());
    const int nMultipleOfEight = RoundUpToNextMultipleOfEight(numberOfSegments);

    Eigen::MatrixXd temporary_lnGammaMolecule = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_PDETemperature = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_SecondPDETemperature = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_PDEVolume = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    
    Eigen::Tensor<double, 4, Eigen::RowMajor> temporary_averageInteractionEnergies;
    Eigen::Tensor<double, 3, Eigen::RowMajor> temporary_partialMolarEnergies;

    if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {

        temporary_averageInteractionEnergies = Eigen::Tensor<double, 4, Eigen::RowMajor>(int(_calculation.concentrations.size()),
            param.numberOfPartialInteractionMatrices + 1, // +1 because A_int is the first one
            int(_calculation.components.size()),
            int(_calculation.components.size()));

        temporary_averageInteractionEnergies.setZero();

        if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
            temporary_partialMolarEnergies = Eigen::Tensor<double, 3, Eigen::RowMajor>(int(_calculation.concentrations.size()),
                param.numberOfPartialInteractionMatrices + 1, // +1 because A_int is the first one
                int(_calculation.components.size()));

            temporary_partialMolarEnergies.setZero();

        }

    }

    double Delta = 0.0001;

    const double dampingFactor = 0.6;
    const double dampingFactorComplement = 1 - dampingFactor;
    const double convergenceThreshhold = 0.000001;   // originally this was 0.0000001;
                                                    // however when using AVX and vectorized code (especially with floats)
                                                    // the accuracy of the calculation drops
                                                    // without increasing this value the gammas do not converge
                                                    // however since the convergence criteria was changed from
                                                    // gamma to log(gamma) the larger convergence threshhold 
                                                    // still leads to numerically very similar results

    const int maximumNumberOfIterations = 50000;
    double newGamma = 0.0;

    MatrixCalcType A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType A_int_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType A_int_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType TauX = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
    MatrixCalcType Tau;
    VectorCalcType vtempSum;
    VectorCalcType PartialSegmentGammasPartialT; // for calculations with modified Hessian
    VectorCalcType PartialSegmentGammasPartialTWithHessian; // for calculations with unmodified Hessian according to eq. (52)
    VectorCalcType PartialSegmentGammasPartialv;
    VectorCalcType Partial2SegmentGammasPartialT2;
    if (param.sw_phi == 1) {
        vtempSum = VectorCalcType::Zero(numberOfSegments);
    }

    std::vector<Eigen::MatrixXd> partialInteractionMatrices;

    if (param.numberOfPartialInteractionMatrices > 0) {
        for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
            partialInteractionMatrices.push_back(Eigen::MatrixXd::Zero(numberOfSegments, numberOfSegments));
        }
    }

    for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {

        // conditions
        double temperature = _calculation.TauTemperatures[g];

#ifdef MEASURE_TIME
        std::chrono::high_resolution_clock::time_point calculateTau_last = std::chrono::high_resolution_clock::now();
#endif
        bool TauIsEmpty = _calculation.Taus[g].size() == 0;
        if (TauIsEmpty || withTDerivative) {
            #pragma omp critical
            {
                Tau = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
                A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                if (withTDerivative) {
                    A_int_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                    A_int_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                }
                else {
                    A_int_PDET = MatrixCalcType::Zero(1, 1);
                    A_int_SecondPDET = MatrixCalcType::Zero(1, 1);
                }

                // calculate interaction matrices
                if (param.numberOfPartialInteractionMatrices > 0) {
                    for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                        partialInteractionMatrices[h].setZero();
                    }
                }

                calculateInteractionMatrix(_calculation.segments, A_int, A_int_PDET, A_int_SecondPDET, partialInteractionMatrices, param, temperature);

                // if the full interaction matrix is needed, fill upper right half
                if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0 || withTDerivative) {
                    for (int i = 0; i < numberOfSegments; i++) {
                        for (int j = i + 1; j < numberOfSegments; j++) {
                            // always j >= i + 1
                            A_int(i, j) = A_int(j, i);
                            if (withTDerivative) {
                                A_int_PDET(i, j) = A_int_PDET(j, i);
                                A_int_SecondPDET(i, j) = A_int_SecondPDET(j, i);
                            }
                        }
                    }
                    for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                        for (int i = 0; i < numberOfSegments; i++) {
                            for (int j = i + 1; j < numberOfSegments; j++) {
                                // always j >= i + 1
                                partialInteractionMatrices[h](i, j) = partialInteractionMatrices[h](j, i);
                            }
                        }
                    }
                }

                // calculate Tau from the interaction matrix A_int
                int idx = -1;
                int columnSum = 0;
                const calcType minus_div_RT = calcType(-1.0f / (R_GAS_CONSTANT * temperature));

                calcType val;
                calcType* Tau_1D = &(Tau(0, 0));
                for (int i = 0; i < numberOfSegments; i++) {
                    columnSum = i * nMultipleOfEight;
                    for (int j = i; j < numberOfSegments; j++) {
                        // always j >= i
                        val = exp(A_int(j, i) * minus_div_RT);

                        idx = columnSum + j;
                        Tau_1D[idx] = val;
                        Tau_1D[j * nMultipleOfEight + i] = val;
                    }
                }
                _calculation.Taus[g] = Tau;
            }
        } else {
            Tau = _calculation.Taus[g];
        }

#ifdef MEASURE_TIME
        calculateTau_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateTau_last).count());
#endif

#ifdef PRINT_DEBUG_INFO
        Eigen::MatrixXd A_int_d = A_int.cast<double>();
        for (int i = 0; i < numberOfSegments; i++)
            for (int j = i; j < numberOfSegments; j++)
                A_int_d(i, j) = A_int_d(j, i);

        WriteEigenMatrixtoFile("A_int_" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int_d); // symmetric

        Eigen::MatrixXd Tau_d = Tau.cast<double>();
        WriteEigenMatrixToFile("Tau_" + std::to_string(_calculation.number) + "_" + std::to_string(g), Tau_d);
#endif


        for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {

            int i = _calculation.TauConcentrationIndices[g][h];

            if (actualConcentrationIndicesToCalculate.size() > 0)
                if (std::find(actualConcentrationIndicesToCalculate.begin(), actualConcentrationIndicesToCalculate.end(), i) == actualConcentrationIndicesToCalculate.end())
                    continue;

            calcType* gammas = &(_calculation.segmentGammas(0, i));
            VectorCalcType SegmentGammas = _calculation.segmentGammas.col(i).head(numberOfSegments);
#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point calculateCOSMOSPACE_last = std::chrono::high_resolution_clock::now();
#endif
            TauX = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
            int lowerBoundIndexForCOSMOSPACECalculation = _calculation.lowerBoundIndexForCOSMOSPACECalculation[i];
            int upperBoundIndexForCOSMOSPACECalculation = _calculation.upperBoundIndexForCOSMOSPACECalculation[i];

            { // as local as possible scope for all variables used in your loops

                const calcType* vTau_1D = &(Tau(0, 0));
                const calcType* vX = &(_calculation.segmentConcentrations(0, i));

                calcType* vTauX_1D = &(TauX(0, 0));
                int idx = 0;
                int columnSum = 0;
                using SimdCalc = Simd<calcType>;


                for (int j = 0; j < numberOfSegments; j++) {

                    columnSum = j * nMultipleOfEight;

                    for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += SimdCalc::lanes) {
                        idx = columnSum + k;
                        SimdCalc::store(
                            vTauX_1D + idx,
                            SimdCalc::mul(SimdCalc::load(vTau_1D + idx), SimdCalc::load(vX + k))
                        );
                    }

                }
            }

#ifdef PRINT_DEBUG_INFO
            Write1DArrayToFile<calcType>("X_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), &(_calculation.segmentConcentrations(0, i)), 1, nMultipleOfEight);
            Eigen::MatrixXd TauX_d = TauX.cast<double>();
            WriteEigenMatrixToFile("TauX_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), TauX_d);
#endif

            int numberOfConvergedGammas = 0;
            int numberOfIteration = 0;

            while (numberOfConvergedGammas != numberOfSegments) {

                numberOfIteration++;

                if (numberOfIteration > maximumNumberOfIterations) {

                    if (param.sw_skip_COSMOSPACE_errors == 0) {

                        std::string information = "";
                        information += "\nSYSTEM_COMPONENTS:\n";
                        for (int j = 0; j < _calculation.components.size(); j++) {
                            information += " -" + _calculation.components[j]->name + "\n";
                        }

                        calcType* Tau_1D = &(Tau(0, 0));
                        for (int j = 0; j < numberOfSegments * nMultipleOfEight; j++) {
                            if (std::isnan(Tau_1D[j])) {
                                information += " Some Tau entries are NaN.";
                                break;
                            }
                        }

                        for (int j = 0; j < numberOfSegments * nMultipleOfEight; j++) {
                            if (std::isinf(Tau_1D[j])) {
                                information += " Some Tau entries are inf.";
                                break;
                            }
                        }

                        for (int j = 0; j < numberOfSegments; j++) {
                            if (std::isnan(_calculation.segmentGammas(j, i))) {
                                information += " Some gammas are NaN.";
                                break;
                            }
                        }
                        for (int j = 0; j < numberOfSegments; j++) {
                            if (std::isinf(_calculation.segmentGammas(j, i))) {
                                information += " Some gammas are inf.";
                                break;
                            }
                        }
                        Eigen::MatrixXd A_int_d = A_int.cast<double>();
                        for (int i = 0; i < numberOfSegments; i++)
                            for (int j = i; j < numberOfSegments; j++)
                                A_int_d(i, j) = A_int_d(j, i);

#ifdef PRINT_DEBUG_INFO
                        WriteEigenMatrixtoFile("A_int_" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int_d); // symmetric

                        Eigen::MatrixXd Tau_d = Tau.cast<double>();
                        WriteEigenMatrixtoFile("Tau_" + std::to_string(_calculation.number) + "_" + std::to_string(g), Tau_d); // symmetric

                        Write1DArraytoFile<double>("X_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), &(_calculation.segmentConcentrations(0, i)), 1, nMultipleOfEight, true);
                        Eigen::MatrixXd TauX_d = TauX.cast<double>();
                        WriteEigenMatrixtoFile("TauX_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), TauX_d, true);
#endif           

                        display(information + "\nCOSMOSPACE did not converge for calculation " + std::to_string(_calculation.number) + " on concentration " + std::to_string(i) + ": maximum number of iterations reached. (" + std::to_string(numberOfConvergedGammas) + "/" + std::to_string(numberOfSegments) + ")");
                    }
                    else {
                        // set all segment gammas to 1 as initial point for next execution
                        _calculation.segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
                    }

                    throw std::runtime_error("COSMOSPACE did not converge for calculation " + std::to_string(_calculation.number) + " on concentration " + std::to_string(i) + ": maximum number of iterations reached. (" + std::to_string(numberOfConvergedGammas) + "/" + std::to_string(numberOfSegments) + ")");
                }

                numberOfConvergedGammas = 0;

                int columnSum = 0;

                for (int j = 0; j < numberOfSegments; j++) {
                    newGamma = 0.0;

                    columnSum = j * nMultipleOfEight;

                    { // as local as possible scope for all variables used in your loops

                        const calcType* vTauX = &(TauX(0, 0));
                        calcType* vGammas = gammas;
                        using SimdCalc = Simd<calcType>;

                        SimdCalc::Vec tempVectorSum = SimdCalc::zero();

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += SimdCalc::lanes) {
                            tempVectorSum = SimdCalc::fmadd(
                                SimdCalc::load(vTauX + columnSum + k),
                                SimdCalc::load(vGammas + k),
                                tempVectorSum
                            );
                        }

                        newGamma = double(SimdCalc::horizontal_sum(tempVectorSum));
                    }

                    newGamma = 1 / newGamma;
                    double oldGamma = double(gammas[j]);

                    // apply damping
                    if (numberOfIteration > 200) {
                        newGamma = dampingFactor * newGamma + dampingFactorComplement * oldGamma;
                    }

                    // check convergence criteria
                    double temp_convergenceThreshhold = convergenceThreshhold;
                    if (j == numberOfSegments - 1) {
                        temp_convergenceThreshhold = 0.0000001;
                    }
                    if (abs(log(newGamma) - log(gammas[j])) <= temp_convergenceThreshhold) {
                        numberOfConvergedGammas++;
                    }

                    gammas[j] = calcType(newGamma);

                    if (param.sw_phi == 1) {
                        SegmentGammas(j) = calcType(newGamma);
                        vtempSum(j) = calcType(1 / newGamma);
                    }
                }
            }

#ifdef PRINT_DEBUG_INFO
            display("COSMOSPACE converged with this number of iterations: " + std::to_string(numberOfIteration) + "\n");
            Write1DArrayToFile<calcType>("Gammas_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), gammas, 1, numberOfSegments, true);
#endif

            if (param.sw_phi == 1 && (withTDerivative == true || withvDerivative == true)) {
                bool iIsReferenceStateIndex = false;
                for (int p = 0; p < _calculation.referenceStateCalculationIndices.size(); p++) {
                    for (int q = 0; q < int(_calculation.components.size() - 1); q++) {
                        if (_calculation.referenceStateCalculationIndices[p][q] == i) {
                            iIsReferenceStateIndex = true;
                            break;
                        }
                    }
                    if (iIsReferenceStateIndex == true) {
                        break;
                    }
                }

                if (iIsReferenceStateIndex == true && withTDerivative == true) {
                    //PartialSegmentGammasPartialT = ((- 1 / (R_GAS_CONSTANT * temperature * temperature) * A_int.col(numberOfSegments - 1)).array() * (A_int.col(numberOfSegments - 1) / (R_GAS_CONSTANT * temperature)).array().exp()).matrix();
                    //Partial2SegmentGammasPartialT2 = ((A_int.col(numberOfSegments - 1) / (R_GAS_CONSTANT * temperature)).array().exp() * (2 / (R_GAS_CONSTANT * pow(temperature, 3)) * A_int.col(numberOfSegments - 1) + 1 / (pow(R_GAS_CONSTANT, 2) * pow(temperature, 4)) * A_int.col(numberOfSegments - 1).cwiseProduct(A_int.col(numberOfSegments - 1))).array()).matrix();
                    
                    const int N = numberOfSegments;

                    const long double R = (long double)R_GAS_CONSTANT;
                    const long double T = (long double)temperature;

                    const long double invRT = 1.0L / (R * T);
                    const long double invRT2 = invRT / T;        // 1/(R*T^2)
                    const long double invRT3 = invRT2 / T;       // 1/(R*T^3)
                    const long double invRT4 = invRT3 / T;       // 1/(R*T^4)
                    const long double invR2T4 = invRT4 / R;      // 1/(R^2*T^4)

                    PartialSegmentGammasPartialT.resize(N);
                    Partial2SegmentGammasPartialT2.resize(N);

                    for (int p = 0; p < N; ++p) {

                        long double A = (long double)A_int(p, numberOfSegments - 1);

                        long double arg = A * invRT;
                        long double expval = expl(arg);

                        long double val =
                            -A * invRT2 * expval;

                        PartialSegmentGammasPartialT(p) = (double)val;

                        long double term1 = 2.0L * A * invRT3;
                        long double term2 = A * A * invR2T4;

                        long double val2 =
                            expval * (term1 + term2);

                        Partial2SegmentGammasPartialT2(p) = (double)val2;
                    }
                    
                    _calculation.segmentGammasPDETemperature.col(i).head(numberOfSegments) = PartialSegmentGammasPartialT;
                    _calculation.segmentGammasSecondPDETemperature.col(i).head(numberOfSegments) = Partial2SegmentGammasPartialT2;
                }
                if (iIsReferenceStateIndex == true && withvDerivative == true) {
                    _calculation.segmentGammasPDEVolume.col(i).head(numberOfSegments).setZero();
                }
                if (iIsReferenceStateIndex == false) {
                    VectorCalcType vg = VectorCalcType::Zero(numberOfSegments);
                    VectorCalcType vg_sum = VectorCalcType::Zero(numberOfSegments);
                    MatrixCalcType Hessian = MatrixCalcType::Zero(numberOfSegments, numberOfSegments); // definition according to eq. (52) in Yan, 2024
                    MatrixCalcType modifiedHessian = MatrixCalcType::Zero(numberOfSegments, numberOfSegments); // definition according to eq. (54) in Yan, 2024
                    MatrixCalcType modifiedHessianPDETemperature = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                    VectorCalcType DeltaGamma = VectorCalcType::Zero(numberOfSegments);

                    for (int p = 0; p < numberOfSegments; p++) {

                        for (int q = p; q < numberOfSegments; q++) {

                            long double xp = (long double)_calculation.segmentConcentrations(p, i);
                            long double xq = (long double)_calculation.segmentConcentrations(q, i);
                            long double tau = (long double)Tau(p, q);

                            long double val_ld = -xp * xq * tau;
                            double val = (double)val_ld;

                            //calcType val2 = -1 * _calculation.segmentConcentrations(p, i) * _calculation.segmentConcentrations(q, i) * A_int(p, q) / (R_GAS_CONSTANT * temperature * temperature) * exp(-1 * A_int(p, q) / (R_GAS_CONSTANT * temperature)); // this is the temperature derivative of the Hessian
                            modifiedHessian(q, p) = (calcType)val;
                            modifiedHessian(p, q) = (calcType)val;
                            Hessian(q, p) = (calcType)val;
                            Hessian(p, q) = (calcType)val;
                            //modifiedHessianPDETemperature(q, p) = (calcType)val2;
                            //modifiedHessianPDETemperature(p, q) = (calcType)val2;
                        }

                    }

                    /* ================================
                        Diagonale der modifizierten Hessian
                    ================================ */
                    // The following code uses vector operations
                    //VectorCalcType constantSecondSummandOfModifiedHessian = (_calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * _calculation.segmentConcentrations.col(i).head(numberOfSegments)).asDiagonal() * Tau.diagonal().head(numberOfSegments);
                    //VectorCalcType tempvec = _calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * vtempSum; 
                    //modifiedHessian.diagonal() = -1 * tempvec.asDiagonal() * SegmentGammas.cwiseInverse() - constantSecondSummandOfModifiedHessian;
                    
                    // This is the computationally heavier version using aggregators
                    for (int p = 0; p < numberOfSegments; ++p) {

                        double xp2 = _calculation.segmentConcentrations(p, i) * _calculation.segmentConcentrations(p, i);
                        long double xp = (long double)_calculation.segmentConcentrations(p, i);
                        long double xpsquared = (long double)xp2;
                        long double gamma_p = (long double)SegmentGammas(p);
                        long double tau_p_p = (long double)Tau(p, p);
                        long double vtempSum_p = (long double)vtempSum(p);

                        long double sum_ld = xpsquared * tau_p_p;

                        long double diag_ld = -(xp * vtempSum_p / gamma_p) - sum_ld;
                        modifiedHessian(p, p) = (calcType)diag_ld;
                    }

                    if (withvDerivative == true) {
                        VectorCalcType dX_dv = _calculation.PartialsegmentConcentrationsPartialv.col(i).head(numberOfSegments);
                        VectorCalcType VectorbForESToDeterminedGamma_dV = VectorCalcType(numberOfSegments);
                        for (int p = 0; p < numberOfSegments; p++) {
                            VectorCalcType Tau_row = Tau.row(p);
                            VectorCalcType CWiseProductSegmentGammasTauRow = SegmentGammas.cwiseProduct(Tau_row);
                            calcType DotProduct = CWiseProductSegmentGammasTauRow.dot(dX_dv);
                            VectorbForESToDeterminedGamma_dV(p) = _calculation.segmentConcentrations(p, i) * DotProduct;
                        }
                        _calculation.segmentGammasPDEVolume.col(i).head(numberOfSegments) = modifiedHessian.ldlt().solve(VectorbForESToDeterminedGamma_dV);
                    }

                    if (withTDerivative == true) {
                        VectorCalcType PartialvgPartialT = VectorCalcType(numberOfSegments);
                        VectorCalcType Partial2vgPartialT2 = VectorCalcType(numberOfSegments);
                        MatrixCalcType Tau_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                        MatrixCalcType Tau_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);

                        const long double Rl = (long double)R_GAS_CONSTANT;
                        const long double Tl = (long double)temperature;

                        const long double invT = 1.0L / Tl;
                        const long double invT2 = invT * invT;
                        const long double invRT2 = invT2 / Rl;

                        for (int p = 0; p < numberOfSegments; p++) {

                            long double xp = (long double)_calculation.segmentConcentrations(p, i);

                            ///// Calculations for Yan's method /////
                            /* ---- dTau / dT ---- */
                            // The following code lines use vector operations
                            //VectorCalcType Tau_row = Tau.row(p);
                            //VectorCalcType dTau_dT = (((A_int.row(p) - temperature * A_int_PDET.row(p)) / (R_GAS_CONSTANT * temperature * temperature)).array() * Tau_row.transpose().array()).matrix().head(numberOfSegments);
                            //Tau_PDET.row(p) = dTau_dT;

                            // This is the more expensive way using aggregators
                            for (int q = 0; q < numberOfSegments; ++q) {

                                long double A = (long double)A_int(p, q);
                                long double A_T = (long double)A_int_PDET(p, q);
                                long double tau = (long double)Tau(p, q);

                                long double dTau = (A - temperature * A_T) / (R_GAS_CONSTANT * temperature * temperature) * tau;

                                Tau_PDET(p, q) = (calcType)dTau;
                            }


                            /* ---- Partial vg / Partial T ---- */
                            // This is the fast way using vector operations
                            //PartialvgPartialT(p) = -1 * _calculation.segmentConcentrations(p, i) * ((_calculation.segmentConcentrations.col(i).head(numberOfSegments).array() * SegmentGammas.array()).matrix().dot(dTau_dT));
                            
                            // This is the more expensive way using aggregators
                            long double acc_ld = 0.0L;
                            for (int q = 0; q < numberOfSegments; ++q) {
                                long double xq = (long double)_calculation.segmentConcentrations(q, i);
                                long double gamma_q = (long double)SegmentGammas(q);
                                long double dTau = (long double)Tau_PDET(p, q);
                                acc_ld += xq * gamma_q * dTau;
                            }

                            PartialvgPartialT(p) = (calcType)(-xp * acc_ld);
                        }


                        ///// Calculations with Yans' Method /////

                        // The following expression for vg will always be 0. This is also theoretically expected, as the gradient vanishes at the solution (the maximum of Q).
                        // However, since a convergence threshold > 0 was chosen for the Gammas, the gradient is technically not exactly 0. 
                        // It would then need to be calculated above in the form g_m = X_m div oldgamma_m - X_m div newgamma_m.
                        // However, since the paper also assumes that g_m = 0 for the modified Hessian, this will also be assumed going forward.
                        //vg = _calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * SegmentGammas.cwiseInverse() - tempvec; // should be zero after convergence of Gammas

                        PartialSegmentGammasPartialT = modifiedHessian.ldlt().solve(-PartialvgPartialT);


                        for (int p = 0; p < numberOfSegments; p++) {
                            long double xp = (long double)_calculation.segmentConcentrations(p, i);

                            /* ---- d2Tau / dT2 ---- */
                            // The following code lines use vector operations
                            //VectorCalcType Tau_row_l = Tau.row(p);
                            //VectorCalcType dTau_dT_l = Tau_PDET.row(p);
                            //VectorCalcType d2Tau_dT2_l = (((-temperature * A_int_SecondPDET.row(p) + 2 * A_int_PDET.row(p) - 2 * A_int.row(p) / temperature) / (R_GAS_CONSTANT * temperature * temperature)).array() * Tau_row_l.transpose().array()).matrix() + (dTau_dT_l.transpose().array() * ((A_int.row(p) - temperature * A_int_PDET.row(p)) / (R_GAS_CONSTANT * temperature * temperature)).array()).matrix();
                            //Tau_SecondPDET.row(p) = d2Tau_dT2_l;

                            // This is the more expensive way using aggregators
                            for (int q = 0; q < numberOfSegments; ++q) {

                                long double A = (long double)A_int(p, q);
                                long double A_T = (long double)A_int_PDET(p, q);
                                long double A_TT = (long double)A_int_SecondPDET(p, q);
                                long double tau = (long double)Tau(p, q);
                                long double dTau = (long double)Tau_PDET(p, q);

                                /* first factor for d2TaudT2 */
                                long double termA = (-temperature * A_TT + 2.0L * A_T - 2.0L * A * invT) * invRT2;

                                /* second factor for d2TaudT2 */
                                long double termB = (A - temperature * A_T) * invRT2;

                                long double d2Tau = tau * termA + dTau * termB;

                                Tau_SecondPDET(p, q) = (double)d2Tau;
                            }

                            /* ---- Partial2 vg / Partial T2 ---- */
                            // This is the fast way using vector operations (not including terms dX/dT and d^2X/dT^2, so evaluated at constant v and x)
                            //VectorCalcType Tau_row = Tau.row(p).head(numberOfSegments).transpose();
                            //VectorCalcType dTau_dT = Tau_PDET.row(p).head(numberOfSegments).transpose();
                            //VectorCalcType d2Tau_dT2 = Tau_SecondPDET.row(p).head(numberOfSegments).transpose();
                            //Partial2vgPartialT2(p) = 2 * _calculation.segmentConcentrations(p, i) / SegmentGammas(p) * PartialSegmentGammasPartialT(p) * (-PartialvgPartialT(p) / _calculation.segmentConcentrations(p, i) + (_calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(Tau_row)).dot(PartialSegmentGammasPartialT)) + _calculation.segmentConcentrations(p, i) * (2 * _calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(dTau_dT).dot(PartialSegmentGammasPartialT) + _calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(SegmentGammas).dot(d2Tau_dT2));
                            
                            // This is the more expensive way using aggregators (not including terms dX/dT and d^2X/dT^2, so evaluated at constant v and x)
                            long double gamma_p = (long double)SegmentGammas(p);
                            long double dGamma_p = (long double)PartialSegmentGammasPartialT(p);

                            long double term1 = 0.0L;
                            long double term2 = 0.0L;

                            for (int q = 0; q < numberOfSegments; ++q) {
                                long double xq = (long double)_calculation.segmentConcentrations(q, i);
                                long double tau = (long double)Tau(p, q);
                                long double dTau = (long double)Tau_PDET(p, q);
                                long double d2Tau = (long double)Tau_SecondPDET(p, q);
                                long double gamma_q = (long double)SegmentGammas(q);
                                long double dGamma_q = (long double)PartialSegmentGammasPartialT(q);

                                term1 += xq * tau * dGamma_q + xq * gamma_q * dTau;
                                term2 += 2 * xq * dGamma_q * dTau + xq * SegmentGammas(q) * d2Tau;
                            }

                            long double val_ld = 2.0L * xp / gamma_p * dGamma_p * term1 + xp * term2;

                            Partial2vgPartialT2(p) = (double)val_ld;
                        }

                        Partial2SegmentGammasPartialT2 = modifiedHessian.ldlt().solve(Partial2vgPartialT2);

                        _calculation.segmentGammasPDETemperature.col(i).head(numberOfSegments) = PartialSegmentGammasPartialT;
                        _calculation.segmentGammasSecondPDETemperature.col(i).head(numberOfSegments) = Partial2SegmentGammasPartialT2;
                    }
                }
            }

#ifdef MEASURE_TIME
            calculateCOSMOSPACE_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateCOSMOSPACE_last).count());
            std::chrono::high_resolution_clock::time_point calculateContactStatistics_last = std::chrono::high_resolution_clock::now();
#endif


            if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                calculateContactStatistics(_calculation, A_int, partialInteractionMatrices, Tau, gammas, int(i), temporary_averageInteractionEnergies, temporary_partialMolarEnergies, param);
            }

#ifdef MEASURE_TIME
            calculateContactStatistics_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateContactStatistics_last).count());
            std::chrono::high_resolution_clock::time_point calculateGammasForMolecules_last = std::chrono::high_resolution_clock::now();
#endif
            double div_Aeff = 1 / param.Aeff;

            for (int j = 0; j < _calculation.components.size(); j++) {

                
                long double lnGammaMolecule = 0.0L;
                long double lnGammaMoleculePDETAtConstantvx = 0.0L;
                long double lnGammaMoleculeSecondPDETAtConstantvx = 0.0L;
                long double lnGammaMoleculePDEvAtConstantTx = 0.0L;

                for (int k = 0; k < numberOfSegments; k++) {
                    long double Akj = (long double)_calculation.segments.SegmentTypeAreas[k][j];
                    long double divaeff = (long double)div_Aeff;
                    long double nkj = Akj * divaeff;
                    long double gamma = (long double)gammas[k];
                    long double ln_gamma = log(gamma);

                    lnGammaMolecule += nkj * ln_gamma; // log(gammas[k]) is calculated more than once although not necessary

                    if (param.sw_phi == 1 && withTDerivative) {
                        long double divgamma = 1 / gamma;
                        long double dgammadT = PartialSegmentGammasPartialT(k);
                        lnGammaMoleculePDETAtConstantvx += nkj * divgamma * dgammadT;

                        long double d2gammadT2 = Partial2SegmentGammasPartialT2(k);
                        lnGammaMoleculeSecondPDETAtConstantvx += nkj * (divgamma * d2gammadT2 - std::pow(divgamma * dgammadT, 2));
                    }
                    if (param.sw_phi == 1 && withvDerivative) {
                        long double divgamma = 1 / gamma;
                        long double dgammadv = _calculation.segmentGammasPDEVolume(k, i);
                        lnGammaMoleculePDEvAtConstantTx += nkj * divgamma * dgammadv;
                    }
                }
                
                temporary_lnGammaMolecule(i, j) = double(lnGammaMolecule);
                temporary_lnGammaMolecule_PDETemperature(i, j) = double(lnGammaMoleculePDETAtConstantvx);
                temporary_lnGammaMolecule_SecondPDETemperature(i, j) = double(lnGammaMoleculeSecondPDETAtConstantvx);
                temporary_lnGammaMolecule_PDEVolume(i, j) = double(lnGammaMoleculePDEvAtConstantTx);

#ifdef PRINT_DEBUG_INFO
                display("lnGammaResidual_1_" + std::to_string(h) + ": " + std::to_string(lnGammaMolecule) + "\n");
#endif

                temporary_lnGammaMolecule(i, j) = lnGammaMolecule;
            }

#ifdef MEASURE_TIME
            calculateGammasForMolecules_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateGammasForMolecules_last).count());
#endif
        }
    }

#ifdef MEASURE_TIME
    std::chrono::high_resolution_clock::time_point calculateGammasForMolecules_last = std::chrono::high_resolution_clock::now();
#endif

    /* Convert to correct reference states. */
    for (int h = 0; h < _calculation.originalNumberOfCalculations; h++) {

        int i = _calculation.actualConcentrationIndices[h];

        if (actualConcentrationIndicesToCalculate.size() > 0)
            if (std::find(actualConcentrationIndicesToCalculate.begin(), actualConcentrationIndicesToCalculate.end(), i) == actualConcentrationIndicesToCalculate.end())
                continue;

        for (int j = 0; j < _calculation.components.size(); j++) {


            double multiplier = 1.0;
            // for an ion with reference state PureComponentsOnlyNeutral
            if (_calculation.referenceStateType[h] == 1 && _calculation.referenceStateCalculationIndices[h][j] == -1) {
                multiplier = NAN;
            }
            _calculation.lnGammaResidual(h, j) = multiplier * temporary_lnGammaMolecule(i, j);
            if (withTDerivative) {
                _calculation.lnGammaResidualPDETatConstantvx(h, j) = multiplier * temporary_lnGammaMolecule_PDETemperature(i, j);
                _calculation.lnGammaResidualSecondPDETatConstantvx(h, j) = multiplier * temporary_lnGammaMolecule_SecondPDETemperature(i, j);
            }
            if (withvDerivative) {
                _calculation.lnGammaResidualPDEvatConstantTx(h, j) = multiplier * temporary_lnGammaMolecule_PDEVolume(i, j);
            }

#ifdef PRINT_DEBUG_INFO
            display("lnGammaResidual_2_" + std::to_string(h) + ": " + std::to_string(_calculation.lnGammaResidual(h, j)) + "\n");
#endif

            if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                for (int k = 0; k < _calculation.components.size(); k++) {
                    _calculation.averageSurfaceEnergies(h, 0, j, k) = multiplier * temporary_averageInteractionEnergies(i, 0, j, k);

                    for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                        _calculation.averageSurfaceEnergies(h, l + 1, j, k) = multiplier * temporary_averageInteractionEnergies(i, l + 1, j, k);
                    }
                }

                if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
                    _calculation.partialMolarEnergies(h, 0, j) = multiplier * temporary_partialMolarEnergies(i, 0, j);

                    for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                        _calculation.partialMolarEnergies(h, l + 1, j) = multiplier * temporary_partialMolarEnergies(h, l + 1, j);
                    }
                }
            }

            // only subtract reference state value if reference state type is not COSMO
            if (_calculation.referenceStateType[h] != 3 && _calculation.referenceStateType[h] != 4) {
                int indexOfReferenceStateCalculation = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[h][j]];

                _calculation.lnGammaResidual(h, j) -= temporary_lnGammaMolecule(indexOfReferenceStateCalculation, j);
                if (withTDerivative) {
                    _calculation.lnGammaResidualPDETatConstantvx(h, j) -= temporary_lnGammaMolecule_PDETemperature(indexOfReferenceStateCalculation, j);
                    _calculation.lnGammaResidualSecondPDETatConstantvx(h, j) -= temporary_lnGammaMolecule_SecondPDETemperature(indexOfReferenceStateCalculation, j);
                }
                if (withvDerivative) {
                    _calculation.lnGammaResidualPDEvatConstantTx(h, j) -= temporary_lnGammaMolecule_PDEVolume(indexOfReferenceStateCalculation, j);
                }


                if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                    for (int k = 0; k < _calculation.components.size(); k++) {
                        _calculation.averageSurfaceEnergies(h, 0, j, k) -= temporary_averageInteractionEnergies(indexOfReferenceStateCalculation, 0, j, k);

                        for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                            _calculation.averageSurfaceEnergies(h, l + 1, j, k) -= temporary_averageInteractionEnergies(indexOfReferenceStateCalculation, l + 1, j, k);
                        }
                    }

                    if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
                        _calculation.partialMolarEnergies(h, 0, j) -= temporary_partialMolarEnergies(indexOfReferenceStateCalculation, 0, j);

                        for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                            _calculation.partialMolarEnergies(h, l + 1, j) -= temporary_partialMolarEnergies(indexOfReferenceStateCalculation, l + 1, j);
                        }
                    }
                }
            }
        }
    }

#ifdef MEASURE_TIME
    calculateGammasForMolecules_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateGammasForMolecules_last).count());
#endif
}

void calculateLnGammaResidual(parameters& param, calculation& _calculation, bool withTDerivative = false, bool withvDerivative = false) {

    std::vector<int> actualConcentrationIndicesToCalculate(0);

    calculateLnGammaResidual(param, _calculation, actualConcentrationIndicesToCalculate, withTDerivative, withvDerivative);
}

// for openCOSMO-RS-phi this could be masde faster by implementing the following line
// and passing it to calculateLnGammaResidual as only some of the concentrations (the ones were the density has not been found)
// need to be recalculated:
// 
// void calculateEOS(parameters& param, calculation& _calculation, const std::vector<int>& originalCalculationsToRun = {}) {
void calculateEOS(parameters& param, calculation& _calculation, int originalConcentrationIndex, bool withTDerivative = false, bool withvDerivative = false, bool DoNotSetGammasToOne = false) {

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[originalConcentrationIndex];
    int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[originalConcentrationIndex][0]];

    int numberOfOriginalComponents = int(_calculation.components.size()) - 1;
    int holeIndex = numberOfOriginalComponents; // it's the last component

    double temperature = double(_calculation.temperatures[actualConcentrationIndex]);
    double RT = R_GAS_CONSTANT * temperature;
    double v = _calculation.molarVolume(originalConcentrationIndex);

    // n_total = 1 -> V = v
    // i.e. volume = molar volume
    double molarVolumeMinusMixtureCavityVolume = std::max(v - _calculation.mixtureCavityVolume(originalConcentrationIndex), 0.0);

    // molesOfHoles can't be negative, its a hardcore and it would lead to negative concentrations
    double molesOfHoles = (1 / _calculation.mixtureHoleVolume(originalConcentrationIndex)) * molarVolumeMinusMixtureCavityVolume;
    double newTotalMols = molesOfHoles + 1; // molesOfHoles + sum of prior n_total
    double holeConcentration = molesOfHoles / newTotalMols;
    double PartialholeConcentrationPartialv = 1 / (_calculation.mixtureHoleVolume(originalConcentrationIndex) * std::pow(newTotalMols, 2));

    if (_calculation.recalculateConcentrationsForPhi) {

        for (int j = 0; j < numberOfOriginalComponents; j++) {
            double newConcentration = _calculation.originalConcentrations(originalConcentrationIndex, j) * (1 - holeConcentration);
            _calculation.concentrations[actualConcentrationIndex][j] = newConcentration;
        }
        _calculation.concentrations[actualConcentrationIndex][numberOfOriginalComponents] = holeConcentration;

        calculateSegmentConcentrations(_calculation, std::vector<int>{actualConcentrationIndex});
        calculatePartialSegmentConcentrationsPartialv(_calculation, std::vector<int>{actualConcentrationIndex}, PartialholeConcentrationPartialv);
        _calculation.recalculateConcentrationsForPhi = true;
    }

    // gamma residual
    if (!DoNotSetGammasToOne) {
        _calculation.segmentGammas(Eigen::indexing::all, actualConcentrationIndex).setOnes();
        _calculation.segmentGammas(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setOnes();
    }
    _calculation.segmentGammasPDETemperature(Eigen::indexing::all, actualConcentrationIndex).setZero();
    _calculation.segmentGammasPDETemperature(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setZero();
    _calculation.segmentGammasPDEVolume(Eigen::indexing::all, actualConcentrationIndex).setZero();
    _calculation.segmentGammasPDEVolume(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setZero();

    calculateLnGammaResidual(param, _calculation, std::vector<int>{actualConcentrationIndex, actualReferenceStateConcentrationIndex}, withTDerivative, withvDerivative);

    // residual chemical potential
    Eigen::VectorXd residualChemicalPotential = _calculation.lnGammaResidual(originalConcentrationIndex, Eigen::indexing::all).cast<double>() * RT;

    // pressures
    _calculation.repulsivePressure(originalConcentrationIndex) = RT / molarVolumeMinusMixtureCavityVolume;
    _calculation.attractivePressure(originalConcentrationIndex) = std::min(-1 * residualChemicalPotential(holeIndex) / (_calculation.mixtureHoleVolume(originalConcentrationIndex)), 0.0);
    _calculation.totalPressure(originalConcentrationIndex) = std::max(_calculation.repulsivePressure(originalConcentrationIndex) + _calculation.attractivePressure(originalConcentrationIndex), 1e-8);

    // fugacity coefficients
    double ln_Z = log(_calculation.totalPressure(originalConcentrationIndex) * v / RT);
    Eigen::VectorXd lnPhiRepulsive = (_calculation.componentCavityVolume.array() / molarVolumeMinusMixtureCavityVolume) + log(v / molarVolumeMinusMixtureCavityVolume);
    
    Eigen::VectorXd hemlholtzEnergyDerivative;
    Eigen::VectorXd bracket;
    if (numberOfOriginalComponents == 1) {
        bracket = _calculation.componentCavityVolume / _calculation.mixtureHoleVolume(originalConcentrationIndex);
        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::seqN(0, numberOfOriginalComponents)) + _calculation.componentCavityVolume * double(_calculation.attractivePressure(originalConcentrationIndex));
    }
    else {
        bracket = (_calculation.componentCavityVolume / _calculation.mixtureHoleVolume(originalConcentrationIndex)) + ((_calculation.componentHoleVolume / _calculation.mixtureHoleVolume(originalConcentrationIndex)).array() - 1).matrix() * molesOfHoles;
        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::seqN(0, numberOfOriginalComponents)) - bracket * residualChemicalPotential(holeIndex);
    }
    Eigen::VectorXd lnPhiAttractive = (hemlholtzEnergyDerivative(Eigen::seqN(0, numberOfOriginalComponents)) / RT);

    _calculation.lnPhiAttractive(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiAttractive.cast<double>();
    _calculation.lnPhiRepulsive(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiRepulsive.cast<double>();
    _calculation.lnPhiTotal(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = (lnPhiRepulsive.array() + lnPhiAttractive.array() - ln_Z).matrix().cast<double>();


    // Partial derivatives of derivative properties (molar residual enthalpy and molar heat capacity at consant pressure)
    if (withTDerivative && withvDerivative) {
        // Initializations and pressure derivatives
        Eigen::VectorXd lnGammaResidualPDETatConstantvx = _calculation.lnGammaResidualPDETatConstantvx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        Eigen::VectorXd lnGammaResidualSecondPDETatConstantvx = _calculation.lnGammaResidualSecondPDETatConstantvx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        Eigen::VectorXd lnGammaResidualPDEvatConstantTx = _calculation.lnGammaResidualPDEvatConstantTx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        double dPdTAtConstantvx = -R_GAS_CONSTANT / (_calculation.mixtureHoleVolume(originalConcentrationIndex)) * (residualChemicalPotential(holeIndex) / RT + temperature * lnGammaResidualPDETatConstantvx(holeIndex)) + R_GAS_CONSTANT / molarVolumeMinusMixtureCavityVolume;
        double dPdvAtConstantTx = -RT / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDEvatConstantTx(holeIndex) - RT / std::pow(molarVolumeMinusMixtureCavityVolume, 2);
        double dvdTAtConstantPx = -(dPdTAtConstantvx) / (dPdvAtConstantTx);

        /* ---- Calculation of h^r(T,v,z) = h^R(T,P,z) ---- */
        double RT2 = R_GAS_CONSTANT * std::pow(temperature, 2);
        double hres = 0.0;
        double cpres = 0.0;
        double sum1forcpres = 0.0;
        double sum2forcpres = 0.0;
        for (int j = 0; j < numberOfOriginalComponents; j++) {
            double originalConcentration = _calculation.originalConcentrations(originalConcentrationIndex, j);
            double subtrahend = _calculation.componentCavityVolume(j) / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex);
            double subtrahend_2 = _calculation.componentCavityVolume(j) / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualSecondPDETatConstantvx(holeIndex);
            double lnGammaResidualPDETDifference = lnGammaResidualPDETatConstantvx(j) - subtrahend;
            double lnGammaResidualSecondPDETDifference = lnGammaResidualSecondPDETatConstantvx(j) - subtrahend_2;
            hres += originalConcentration * lnGammaResidualPDETDifference;
            sum1forcpres += originalConcentration * lnGammaResidualPDETDifference;
            sum2forcpres += originalConcentration * lnGammaResidualSecondPDETDifference;
        }
        hres *= -1 * RT2;
        hres = hres - RT2 * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex);
        hres = hres + _calculation.totalPressure(originalConcentrationIndex)  * v - RT;
        _calculation.molarResidualEnthalpy(originalConcentrationIndex) = hres;
        
        /* ---- Calculation of c_p^r(T,v,z) ---- */
        cpres = -2 * RT * sum1forcpres - RT2 * sum2forcpres;
        cpres = cpres - 2 * RT * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex) - RT2 * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualSecondPDETatConstantvx(holeIndex);
        double dPdTAtConstantvxSquared = std::pow(dPdTAtConstantvx, 2);
        cpres = cpres - R_GAS_CONSTANT - temperature * dPdTAtConstantvxSquared / dPdvAtConstantTx;
        _calculation.molarResidualHeatCapacityAtConstantPressure(originalConcentrationIndex) = cpres;
    }
}


/**
 * @brief This function calculates the full isotherm for specified T (pure component) or specified T, z (mixture)
 * @param manual_endP: allows to modify end pressure of isotherm which is by default 10. However, in some cases, smaller P values have to be calculated.
 */
std::vector<std::vector<double>> calculateIsotherm(calculation _calculation, int i_calculation, double manual_endP = 10.0) {

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double pressureGuess = _calculation.targetPressure(i_calculation);

    double current_v = 1.05 * _calculation.mixtureCavityVolume(i_calculation);
    double current_P = 11.0;
    
    // This section ensures that code doesn't start with liquid_P = 1e-8
    _calculation.molarVolume(i_calculation) = current_v;
    calculateEOS(param, _calculation, i_calculation);
    current_P = _calculation.totalPressure(i_calculation);
    if (current_P == 1e-8) {
        int counter = 0;
        while (current_P == 1e-8 && counter < 11) {
            current_v = (1.0 + 0.05 / std::pow(1.5, counter)) * _calculation.mixtureCavityVolume(i_calculation);
            _calculation.molarVolume(i_calculation) = current_v;
            calculateEOS(param, _calculation, i_calculation);
            current_P = _calculation.totalPressure(i_calculation);
            counter += 1;
        }
    }
    current_v *= 1 / 1.001;

    double current_dPdv = 0.0;
    double last_P = std::numeric_limits<double>::max();
    double last_v = 0.0;

    std::vector<double> liquid_v = std::vector<double>();
    std::vector<double> liquid_P = std::vector<double>();
    double liquid_minimum_P = std::numeric_limits<double>::max();

    std::vector<double> vapor_v = std::vector<double>();
    std::vector<double> vapor_P = std::vector<double>();

    std::vector<double> twophaseregion_v = std::vector<double>();
    std::vector<double> twophaseregion_P = std::vector<double>();

    std::vector<double> v = std::vector<double>();
    std::vector<double> P = std::vector<double>();
    std::vector<double> P_repulsive = std::vector<double>();
    std::vector<double> P_attractive = std::vector<double>();
    std::vector<double> dPdv = std::vector<double>();

    double div_Aeff = 1 / param.Aeff;
    int numberOfOriginalComponents = _calculation.components.size() - 1;
    const int numberOfSegments = int(_calculation.segments.size());


    std::string mode = "liquid";
    double end_P = 10.0;
    double multiplier = 1.001;
    double max_v = 1e8;
    bool has_consecutive_outlier = false;
    int outlier_count = 0;
    bool is_outlier_last_value = false;
    while (true) {
        current_v *= multiplier;
        if ((mode == "vapor" && current_P < end_P) || current_v >= max_v)
            break;

        _calculation.molarVolume(i_calculation) = current_v;
        calculateEOS(param, _calculation, i_calculation);
        current_P = _calculation.totalPressure(i_calculation);

        Eigen::VectorXd residualChemicalPotential = _calculation.lnGammaResidual(i_calculation, Eigen::indexing::all).cast<double>() * R_GAS_CONSTANT * double(_calculation.temperatures[_calculation.actualConcentrationIndices[i_calculation]]); // not necessary
        P_attractive.push_back(_calculation.attractivePressure(i_calculation));
        P_repulsive.push_back(_calculation.repulsivePressure(i_calculation));

        v.push_back(current_v);
        P.push_back(current_P);

        
        if (mode == "liquid") {
            if (current_P < last_P) {
                liquid_v.push_back(current_v);
                liquid_P.push_back(current_P);
            }
            else {
                calculateEOS(param, _calculation, i_calculation, false, false, false);
                if (current_P != 1e-8 && last_P != 1e-8) {
                    calculateEOS(param, _calculation, i_calculation, false, true, false);
                    double sum = 0;
                    for (int k = 0; k < numberOfSegments; k++) {
                        double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                        sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                    }
                    current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - _calculation.mixtureCavityVolume(i_calculation), 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                }
                else {
                    current_dPdv = 5; // random
                }

                if (current_dPdv > 0 || (is_outlier_last_value && has_consecutive_outlier)) {
                    mode = "search_vapor_start";
                    liquid_minimum_P = last_P;
                    multiplier = 1.01;
                    outlier_count = 0;
                    has_consecutive_outlier = false;
                    is_outlier_last_value = false;
                }
            }
        }

        if (mode == "search_vapor_start") {
            if (current_P < last_P) {
                calculateEOS(param, _calculation, i_calculation, false, true, false);
                double sum = 0;
                for (int k = 0; k < numberOfSegments; k++) {
                    double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                    sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                }
                current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - _calculation.mixtureCavityVolume(i_calculation), 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                
                if (current_dPdv < 0) {
                    mode = "vapor";
                    if (int(_calculation.components.size()) == 2) {
                        if (manual_endP != 10.0) {
                            end_P = manual_endP; // This case becomes active if in pure component psat calculations an estimated pressure is below 10 Pa so that the isotherm is recalculated
                        }
                        else {
                            end_P = std::max(liquid_minimum_P, 10.0);
                        }
                    }
                    else if (int(_calculation.components.size()) > 2) {
                        if (manual_endP < 10.0) {
                            end_P = manual_endP; // In this case, manual_endP was not provided as the lastP of a different isotherm, but to mark that isotherm calculation should not stop at P = 10 Pa
                        }
                        else {
                            end_P = std::max(manual_endP, 10.0); // manual_endP gives the possibility to provide lastP of a different isotherm
                        }
                    }
                    multiplier = 1.1;
                }
            }
            else if (param.sw_isotherm == 1) {
                twophaseregion_v.push_back(current_v);
                twophaseregion_P.push_back(current_P);
            }
        }

        if (mode == "vapor") {
            if (current_P < last_P) {
                if (current_P > end_P) {
                    vapor_v.push_back(current_v);
                    vapor_P.push_back(current_P);
                    has_consecutive_outlier = false;
                    is_outlier_last_value = false;
                }
            }
            else {
                std::cout << "Have outlier, current_P: " << current_P << std::endl;
                std::cout << "Have outlier, last_P: " << last_P << std::endl;
                std::cout << "Have outlier, current_v: " << current_v << std::endl;
                std::cout << "Have outlier, last_v: " << last_v << std::endl;
                calculateEOS(param, _calculation, i_calculation, false, true, false);
                double sum = 0;
                for (int k = 0; k < numberOfSegments; k++) {
                    double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                    sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                }
                current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - _calculation.mixtureCavityVolume(i_calculation), 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                std::cout << "Have outlier, current_dPdv: " << current_dPdv << std::endl;

                if (current_dPdv > 0) {
                    outlier_count += 1;
                    is_outlier_last_value = true;
                }
                else {
                    has_consecutive_outlier = false;
                }
                if (has_consecutive_outlier == false && outlier_count < 5) {
                    has_consecutive_outlier = true;
                }
                else {
#ifdef DEBUG_INFO
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_v.txt", &v[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P.txt", &P[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_v.txt", &vapor_v[0], vapor_v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_P.txt", &vapor_P[0], vapor_P.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_v.txt", &liquid_v[0], liquid_v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_P.txt", &liquid_P[0], liquid_P.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_attractive.txt", &P_attractive[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_repulsive.txt", &P_repulsive[0], v.size(), 1);
                    calculateEOS(param, _calculation, i_calculation);
#endif
                    display("The pressure of the vapor phase increased with the molar volume. This should not happen.");
                    throw std::runtime_error("The pressure of the vapor phase increased with the molar volume. This should not happen.");
                }
            }
        }
        last_P = current_P;
        last_v = current_v;
    }

    std::vector<std::vector<double>> returnVector;
    returnVector.push_back(v);
    returnVector.push_back(P);
    returnVector.push_back(liquid_v);
    returnVector.push_back(liquid_P);
    returnVector.push_back(vapor_v);
    returnVector.push_back(vapor_P);
    returnVector.push_back(P_repulsive);
    returnVector.push_back(P_attractive);
    if (param.sw_isotherm == 1) {
        returnVector.push_back(twophaseregion_v);
        returnVector.push_back(twophaseregion_P);
    }

#ifdef DEBUG_INFO
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_v.txt", &v[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P.txt", &P[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_v.txt", &vapor_v[0], vapor_v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_P.txt", &vapor_P[0], vapor_P.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_v.txt", &liquid_v[0], liquid_v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_P.txt", &liquid_P[0], liquid_P.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_attractive.txt", &P_attractive[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_repulsive.txt", &P_repulsive[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_bh.txt", &bh[0], bh.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_mures.txt", &mu_res[0], mu_res.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_mixture_cavity_volume.txt", &mixture_cavity_volume[0], mixture_cavity_volume.size(), 1);
    calculateEOS(param, _calculation, i_calculation);
#endif

    return returnVector;

}


/**
 * @brief Compute vapor- and liquid-phase EOS states for a given T-P condition and evaluate the corresponding phase-equilibrium residual.
 *
 * The function solves for molar volumes of both phases at the current pressure guess,
 * computes fugacity coefficients, and optionally updates the pressure guess iteratively
 * until an equilibrium pressure bracket is obtained. If no vapor branch exists, the
 * state is treated as supercritical and phase results are reset.
 * 
 * @param bool: withIterationForPressure:
 * - false: evaluate EOS / fugacity error once at current targetPressure (no pressure update, no isotherm recalculation)
 * - true: iteratively update pressure based on fugacity mismatch until an equilibrium pressure bracket is found
 *   (incl. optional isotherm recomputation and final ITP refinement), used for pure components only
 */
void calculateEqCondForGivenTPn(calculation _calculation, int i_calculation, std::vector<double> &vapor_v, std::vector<double> &vapor_P, std::vector<double> &liquid_v, std::vector<double> &liquid_P, bool withIterationForPressure, bool WritePressureGuessIntoTargetPressure = false) {
    std::vector<double> pressureGuessesTried = std::vector<double>();
    std::vector<double> OF_P_Tried = std::vector<double>();
    bool isSupercriticalIsotherm = vapor_v.size() == 0; // this means supercritical isotherm

    int i_calculation_vapor = i_calculation;
    int i_calculation_liquid = i_calculation + int(_calculation.originalNumberOfCalculations / 2);

    double min_P = 0.0;
    double max_P = 0.0;
    _calculation.helperValue = -1;
    double pressureGuess = _calculation.targetPressure(i_calculation);

    if (isSupercriticalIsotherm) {
        _calculation.molarVolume(i_calculation_vapor) = 0.0;
        _calculation.molarVolume(i_calculation_liquid) = 0.0;

        _calculation.repulsivePressure(i_calculation_vapor) = 0.0;
        _calculation.repulsivePressure(i_calculation_liquid) = 0.0;

        _calculation.attractivePressure(i_calculation_vapor) = 0.0;
        _calculation.attractivePressure(i_calculation_liquid) = 0.0;

        _calculation.totalPressure(i_calculation_vapor) = 0.0;
        _calculation.totalPressure(i_calculation_liquid) = 0.0;

        _calculation.lnPhiAttractive(i_calculation_vapor) = NAN;
        _calculation.lnPhiAttractive(i_calculation_liquid) = NAN;

        _calculation.lnPhiRepulsive(i_calculation_vapor) = NAN;
        _calculation.lnPhiRepulsive(i_calculation_liquid) = NAN;

        _calculation.lnPhiTotal(i_calculation_vapor) = NAN;
        _calculation.lnPhiTotal(i_calculation_liquid) = NAN;
        return;
    } else {
        if (withIterationForPressure == true) {
            min_P = liquid_P[liquid_P.size() - 1];
            max_P = vapor_P[0];
            pressureGuess = (min_P + max_P) / 2;
        }
    }

    bool did_not_converge = false;
    auto f_pressure = [&_calculation, i_calculation_vapor, &vapor_v, &vapor_P, i_calculation_liquid, &liquid_v, &liquid_P, &did_not_converge](double pressureGuess_) {

        auto f_liquid = [&_calculation, i_calculation_liquid, pressureGuess_](double molarVolume) {
            _calculation.molarVolume(i_calculation_liquid) = molarVolume;
            calculateEOS(param, _calculation, i_calculation_liquid);
            double diff = _calculation.totalPressure(i_calculation_liquid) - pressureGuess_;
            //display(std::to_string(molarVolume * 1000) + "  " \
                                                    + std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "  " + std::to_string(1000 * _calculation.lnGammaResidual(i_calculation_liquid, 1)) + "\n");
            return diff;
        };

        auto f_vapor = [&_calculation, i_calculation_vapor, pressureGuess_](double molarVolume) {
            _calculation.molarVolume(i_calculation_vapor) = molarVolume;
            calculateEOS(param, _calculation, i_calculation_vapor);
            double diff = _calculation.totalPressure(i_calculation_vapor) - pressureGuess_;
            //display(std::to_string(molarVolume * 1000) + "  " + \
                                                    std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + "  "\
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_vapor)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "  " + std::to_string(1000 * _calculation.lnGammaResidual(i_calculation_vapor, 1)) + "\n");
            return diff;
        };

        // liquid phase calculation
        int i_v_lower = -1;
        int i_v_upper = -1;
        for (int i_v = 0; i_v < liquid_v.size(); i_v++) {
            if (liquid_P[i_v] < pressureGuess_) {
                i_v_upper = i_v;
                i_v_lower = i_v - 1;
                break;
            }
        }

        if (i_v_lower == -1 || i_v_upper == -1) {
            did_not_converge = true;
            return 1000.0;
        }

        //display("liquid\n");
        double molarVolumeThresholdLiquid = 1e-17;
        double molarVolumeLiquidGuess = ITP(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid, 0.1);

        _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
        calculateEOS(param, _calculation, i_calculation_liquid);

        double diff = abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_);
        if (diff > 3) {
            molarVolumeLiquidGuess = bisection(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid);
            _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
            calculateEOS(param, _calculation, i_calculation_liquid);
            double diffnew = abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_); // for debugging
            if (abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_) > diff) {
                molarVolumeLiquidGuess = ITP(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid, 0.1);
                _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
                calculateEOS(param, _calculation, i_calculation_liquid);
            }
            //display(std::to_string(molarVolumeLiquidGuess * 1000) + "  " + \
                                                    std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + "  "\
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "\n");
        }

        // vapor phase calculation
        i_v_lower = -1;
        i_v_upper = -1;
        for (int i_v = 0; i_v < vapor_v.size(); i_v++) {
            if (vapor_P[i_v] < pressureGuess_) {
                i_v_upper = i_v;
                i_v_lower = i_v - 1;
                break;
            }
        }

        if (i_v_lower == -1 || i_v_upper == -1) {
            did_not_converge = true;
            return 1000.0;
        }

        //display("vapor\n");
        double molarVolumeThresholdVapor = 1e-14;
        if (_calculation.calculationType == "LL") {
            molarVolumeThresholdVapor = 1e-17;
        }
        double molarVolumeVaporGuess = ITP(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor, 1);

        _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
        calculateEOS(param, _calculation, i_calculation_vapor);

        diff = abs(_calculation.totalPressure(i_calculation_vapor) - pressureGuess_);
        if (diff > 3) {
            molarVolumeVaporGuess = bisection(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor);
            _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
            calculateEOS(param, _calculation, i_calculation_vapor);
            if (abs(_calculation.totalPressure(i_calculation_vapor) - pressureGuess_) > diff) {
                molarVolumeVaporGuess = ITP(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor, 1);
                _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
                calculateEOS(param, _calculation, i_calculation_vapor);
                //display(std::to_string(molarVolumeVaporGuess * 1000) + "  " + \
                                                        std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + "  "\
                                                        + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + "  " \
                                                        + std::to_string(_calculation.totalPressure(i_calculation_vapor)) + "  " \
                                                        + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "\n");
            }
        }

        double returnValue = 0.0;
        double error = 0.0;

        for (int i_c = 0; i_c < _calculation.components.size() - 1; i_c++) {
            double vaporConcentration = _calculation.originalConcentrations(i_calculation_vapor, i_c);
            double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation_vapor, i_c));
            double liquidConcentration = _calculation.originalConcentrations(i_calculation_liquid, i_c);
            double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation_liquid, i_c));
            if (abs((fugacityCoefficientVapor - fugacityCoefficientLiquid) / fugacityCoefficientLiquid) > error) {
                error = abs((fugacityCoefficientVapor - fugacityCoefficientLiquid) / fugacityCoefficientLiquid);
                returnValue = vaporConcentration * fugacityCoefficientVapor - liquidConcentration * fugacityCoefficientLiquid;
                _calculation.helperValue = i_c;
            }
        }

        return returnValue;

    };

    double fugacityError = 1000;
    double P_lower = -1;
    double P_upper = -1;
    double OF_P_lower = -1;
    double OF_P_upper = -1;
    double fugacityConvergenceCriteria = 0.0001;

    while (fugacityError > fugacityConvergenceCriteria) {
        pressureGuessesTried.push_back(pressureGuess);
        if (WritePressureGuessIntoTargetPressure == true) {
            _calculation.targetPressure(i_calculation_vapor) = pressureGuess;
            _calculation.targetPressure(i_calculation_liquid) = pressureGuess;
        }
        double OF_P = f_pressure(pressureGuess);
        int decisiveIndex = _calculation.helperValue;
        if (decisiveIndex == -1) {
            did_not_converge = true;
        }
        if (did_not_converge) {
            break;
        }
        double fugacityVapor = _calculation.originalConcentrations(i_calculation_vapor, decisiveIndex) * exp(_calculation.lnPhiTotal(i_calculation_vapor, decisiveIndex));
        double fugacityLiquid = _calculation.originalConcentrations(i_calculation_liquid, decisiveIndex) * exp(_calculation.lnPhiTotal(i_calculation_liquid, decisiveIndex));
        fugacityError = abs(OF_P / fugacityLiquid);

        OF_P_Tried.push_back(OF_P);
        if (OF_P < 0) {
            P_lower = pressureGuess;
            OF_P_lower = OF_P;
        }
        else if (OF_P > 0) {
            P_upper = pressureGuess;
            OF_P_upper = OF_P;
        }


        if (P_lower != -1 && P_upper != -1) {
            break;
        }


        if (withIterationForPressure == false) {
            break;
        }
        else {
            pressureGuess *= (fugacityLiquid / fugacityVapor);
            if (pressureGuess < vapor_P[vapor_P.size() - 1]) {
                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation, std::min(pressureGuess - 1.0, 0.1));
                
                liquid_v = returnVector[2];
                liquid_P = returnVector[3];
                vapor_v = returnVector[4];
                vapor_P = returnVector[5];
            }
        }

        if (pressureGuessesTried.size() > 100) {

            display(std::to_string(_calculation.molarVolume(i_calculation_liquid) * 1000) + " "\
                + std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + " "\
                + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiRepulsive(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiAttractive(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiTotal(i_calculation_liquid)) + "  " \
                + std::to_string(_calculation.molarVolume(i_calculation_vapor) * 1000) + " "\
                + std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + " "\
                + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiRepulsive(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiAttractive(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiTotal(i_calculation_vapor)) + " " \
                + std::to_string(pressureGuess) + "\n");

        }

        if (pressureGuessesTried.size() > 250) {
            //display("did not converge");
            did_not_converge = true;
            break;
        }
    }

    if (!did_not_converge) {
        if (P_lower != -1 && P_upper != -1) {
            pressureGuess = ITP(f_pressure, P_lower, P_upper, OF_P_lower, OF_P_upper, 0.01, fugacityConvergenceCriteria);

            if (WritePressureGuessIntoTargetPressure == true) {
                _calculation.targetPressure(i_calculation_vapor) = pressureGuess;
                _calculation.targetPressure(i_calculation_liquid) = pressureGuess;
            }
        }
    }

    if (did_not_converge) {
        _calculation.molarVolume(i_calculation_vapor) = 0.0;
        _calculation.molarVolume(i_calculation_liquid) = 0.0;

        _calculation.repulsivePressure(i_calculation_vapor) = 0.0;
        _calculation.repulsivePressure(i_calculation_liquid) = 0.0;

        _calculation.attractivePressure(i_calculation_vapor) = 0.0;
        _calculation.attractivePressure(i_calculation_liquid) = 0.0;

        _calculation.totalPressure(i_calculation_vapor) = 0.0;
        _calculation.totalPressure(i_calculation_liquid) = 0.0;

        _calculation.lnPhiAttractive(i_calculation_vapor) = NAN;
        _calculation.lnPhiAttractive(i_calculation_liquid) = NAN;

        _calculation.lnPhiRepulsive(i_calculation_vapor) = NAN;
        _calculation.lnPhiRepulsive(i_calculation_liquid) = NAN;

        _calculation.lnPhiTotal(i_calculation_vapor) = NAN;
        _calculation.lnPhiTotal(i_calculation_liquid) = NAN;
    }
}


double calculatevForGivenTPn(calculation _calculation, int i_calculation, std::vector<double> v_vector, std::vector<double> P_vector, double pressure_) {
    auto f = [&_calculation, i_calculation, pressure_](double molarVolume) {
        _calculation.molarVolume(i_calculation) = molarVolume;
        calculateEOS(param, _calculation, i_calculation);
        double diff = _calculation.totalPressure(i_calculation) - pressure_;
        return diff;
    };

    int i_v_lower = -1;
    int i_v_upper = -1;
    for (int i_v = 0; i_v < v_vector.size(); i_v++) {
        if (P_vector[i_v] < pressure_) {
            i_v_upper = i_v;
            i_v_lower = i_v - 1;
            break;
        }
    }

    if (i_v_lower == -1 || i_v_upper == -1) {
        return 1000.0;
    }

    double molarVolumeLiquidGuess = ITP(f, v_vector[i_v_lower], v_vector[i_v_upper], P_vector[i_v_lower] - pressure_, P_vector[i_v_upper] - pressure_, 1e-16, 0.1);

    return molarVolumeLiquidGuess;
}


#include "phase_equilibrium.hpp";


void calculate(std::vector<int>& calculationIndices) {

    // this is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section ends
    threadException e;
#if defined(_OPENMP)
    omp_set_nested(1);
#pragma omp parallel for
#endif
    for (int i = 0; i < calculationIndices.size(); i++) {

        e.run([=] {
            //#if defined(_OPENMP)
            //int outer_tid = omp_get_thread_num();
            //std::cout << "Outer thread " << outer_tid << "\n";
            //#endif

            int calculationIndex = calculationIndices[i];

#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
            if (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) {

                calculations[calculationIndex].segments.clear();

                for (int j = 0; j < calculations[calculationIndex].components.size(); j++) {

                    std::shared_ptr<molecule> thisMolecule = calculations[calculationIndex].components[j];

                    for (int k = 0; k < thisMolecule->segments.size(); k++) {
                        calculations[calculationIndex].segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
                            thisMolecule->segments.SegmentTypeSigma[k],
                            thisMolecule->segments.SegmentTypeSigmaCorr[k],
                            thisMolecule->segments.SegmentTypeHBtype[k],
                            thisMolecule->segments.SegmentTypeAtomicNumber[k],
                            thisMolecule->segments.SegmentTypeAreas[k][0],
                            thisMolecule->segments.SegmentTypeMoleculeIndex[k]);
                    }
                }
                calculations[calculationIndex].segments.sort();
                calculations[calculationIndex].segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()));
                calculations[calculationIndex].PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()));
            }


            if (param.sw_alwaysCalculateSizeRelatedParameters == 1 || (param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) \
                || (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {

                rescaleSegments(param, calculations[calculationIndex]);
                calculateSegmentConcentrations(calculations[calculationIndex]);
            }

#ifdef PRINT_DEBUG_INFO
            display("number of components: " + std::to_string(calculations[calculationIndex].components.size()) + "\n");
            display("number of segments: " + std::to_string(calculations[calculationIndex].segments.size()) + "\n");
            display("number of concentrations: " + std::to_string(calculations[calculationIndex].concentrations.size()) + "\n");
#endif

#ifdef MEASURE_TIME
            rescaleSegments_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count());
            std::chrono::high_resolution_clock::time_point calculateCombinatorial_last = std::chrono::high_resolution_clock::now();
#endif

            if (param.sw_phi == 1) {
                // Calculations with EoS openCOSMO-RS-Phi

                calculation _calculation = calculations[calculationIndex];

                int numberOfOriginalComponents = _calculation.components.size() - 1;

                _calculation.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);

                for (int i = 0; i < numberOfOriginalComponents; i++) {
                    int index = _calculation.components[i]->index;
                    _calculation.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                    _calculation.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                }

                if (numberOfOriginalComponents == 1) {
                    // Pure Component Calculations

                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                    if (_calculation.originalNumberOfCalculations % 2 != 0)
                        throw std::runtime_error("Both phases are expected to be given as concentrations.");

                    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);

                    //#if defined(_OPENMP)
                    //#pragma omp parallel for
                    //#endif
                    _calculation.invalidateTaus();//recalcualte Taus
                    double smallestTemperatureWhichIsSupercritic = std::numeric_limits<double>::max();

                    #if defined(_OPENMP)
                    #pragma omp parallel for
                    #endif
                    for (int i_calculation = 0; i_calculation < numberOfStatePoints; i_calculation++) {

                        //#if defined(_OPENMP)
                        //int inner_tid = omp_get_thread_num();
                        //std::cout << "  Inner thread " << inner_tid << "\n";
                        //#endif
                        
                        if (param.sw_criticalPoint == 1) {
                            findCriticalPointForSpecifiedx(_calculation, i_calculation);
                        }
                        else {
                            // In the else case it is simply assumed that a pure component VLE calculation is to be carried out
                            double pressureGuess = _calculation.targetPressure(i_calculation);

                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];

                            std::vector<double> v;
                            std::vector<double> P;
                            std::vector<double> liquid_v;
                            std::vector<double> liquid_P;
                            std::vector<double> vapor_v;
                            std::vector<double> vapor_P;
                            std::vector<double> P_repulsive;
                            std::vector<double> P_attractive;

                            if (temperature < smallestTemperatureWhichIsSupercritic) {
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);

                                v = returnVector[0];
                                P = returnVector[1];
                                liquid_v = returnVector[2];
                                liquid_P = returnVector[3];
                                vapor_v = returnVector[4];
                                vapor_P = returnVector[5];
                                P_repulsive = returnVector[6];
                                P_attractive = returnVector[7];
                                bool isSupercriticalIsotherm = vapor_v.size() == 0;

                                if (isSupercriticalIsotherm && temperature < smallestTemperatureWhichIsSupercritic)
                                    smallestTemperatureWhichIsSupercritic = temperature;
                            }

                            calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true, true);
                            if (param.sw_calculateResidualProperties == 1) {
                                calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, true, true);
                                calculateEOS(param, _calculation, i_calculation, true, true);
                            }
                        }
                    }
                }
                else {
                    // Mixture Calculations

                    if (param.sw_criticalPoint_specifiedx == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_Pbub == 1 || param.sw_Tbub == 1 || param.sw_Pdew == 1 || param.sw_Tdew == 1 || param.sw_PT_Flash == 1 || param.sw_excess == 1 || param.sw_lnGamma == 1) {
                        // If one of the switches for phase equilibrium / energetic properties is set, the respective calculations are performed. 
                        performPhaseEquilibriumCalculations(param, _calculation);
                    }
                    else {
                        // If just sw.phi = 1 is set and no other switches are active, a simple evaluation of the EoS for the given T, P, z conditions is performed - once with the liquid-like root and once with the vapor-phase root.
                        int numberOfOriginalComponents = int(_calculation.components.size()) - 1;
                        _calculation.overallConcentrations = _calculation.originalConcentrations.cast<double>();

                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;
                        
                        bool withTDerivative = false;
                        bool withvDerivative = false;
                        if (param.sw_calculateResidualProperties == 1) {
                            withTDerivative = true;
                            withvDerivative = true;
                        }

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            double pressure = _calculation.targetPressure(i_calculation);

                            ///// Calculate Fugacity coefficient (liquid-like root) of all components in mixture
                            std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                            std::vector<double> liquid_v = returnVector[2];
                            std::vector<double> liquid_P = returnVector[3];
                            std::vector<double> vapor_v = returnVector[4];
                            std::vector<double> vapor_P = returnVector[5];

                            double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
                            _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                            calculateEOS(param, _calculation, i_calculation + numberOfCalculations, withTDerivative, withvDerivative);

                            v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                            _calculation.molarVolume(i_calculation) = v;
                            calculateEOS(param, _calculation, i_calculation, withTDerivative, withvDerivative);
                        }
                    }
                }
            }
            else {
                // recalculate combinatorial term if needed
                if ((param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) || param.sw_alwaysCalculateSizeRelatedParameters == 1 || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {
                    calculateLnGammaCombinatorial(param, calculations[calculationIndex]);;
                }

#ifdef MEASURE_TIME
                calculateCombinatorial_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateCombinatorial_last).count());
                std::chrono::high_resolution_clock::time_point calculateResidual_last = std::chrono::high_resolution_clock::now();
#endif

                // calculate residual part
                calculateLnGammaResidual(param, calculations[calculationIndex]);

#ifdef MEASURE_TIME
                calculateResidual_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateResidual_last).count());
#endif

                calculations[calculationIndex].lnGammaTotal = calculations[calculationIndex].lnGammaCombinatorial + calculations[calculationIndex].lnGammaResidual;
#ifdef PRINT_DEBUG_INFO
                for (int i_concentration = 0; i_concentration < calculations[calculationIndex].originalNumberOfCalculations; i_concentration++) {
                    for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                        display("lnGammaTotal_" + std::to_string(calculationIndex) + "_" + std::to_string(i_concentration) + "_" + std::to_string(i_component) + \
                            ": " + std::to_string(calculations[calculationIndex].lnGammaTotal(i_concentration, i_component)) + "\n");
                    }
                }
#endif

                bool calculateSolvationEnergies = param.dGsolv_E_gas.size() > 0;
                if (calculateSolvationEnergies) {
                    double kcalPerMol_per_Hartree = 2625.499639479 / 4.184;
                    double reference_pressure = 101325; // Pa = 1 atm;
                    std::vector<int> atomicNumbersWithout_dGsolv_tau = std::vector<int>();
                    double approximate_dGsolv_tau = 0.0262; // median of other values
                    for (int i_concentration = 0; i_concentration < calculations[calculationIndex].originalNumberOfCalculations; i_concentration++) {
                        if (calculations[calculationIndex].referenceStateType[i_concentration] == 4) {
                            int i_solvent_component = -1;
                            for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                                if (calculations[calculationIndex].concentrations[i_concentration][i_component] == 1.0f) {
                                    i_solvent_component = i_component;
                                    break;
                                }
                            }
                            for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                                double dGsolv = 0.0;
                                if (calculations[calculationIndex].concentrations[i_concentration][i_component] == 0.0f) {

                                    double RT = R_GAS_CONSTANT * calculations[calculationIndex].temperatures[i_concentration];
                                    double molar_volume_ideal_gas = RT / reference_pressure;
                                    double RT_kcalPerMol = RT / (1000 * 4.184);

                                    // all energies calculated below this line are in kcal/mol
                                    double E_vdw = 0.0;
                                    segmentTypeCollection segments = calculations[calculationIndex].components[i_component]->segments;
                                    std::unordered_map<int, double> areasByAtomicNumber;
                                    for (int i_segment = 0; i_segment < segments.size(); i_segment++) {
                                        int AN = segments.SegmentTypeAtomicNumber[i_segment];

                                        if (areasByAtomicNumber.find(AN) == areasByAtomicNumber.end())
                                            areasByAtomicNumber[AN] = 0.0;

                                        areasByAtomicNumber[AN] += segments.SegmentTypeAreas[i_segment][0];
                                    }

                                    for (auto& it : areasByAtomicNumber) {
                                        double this_atom_dGsolv_tau = abs(param.dGsolv_tau[it.first]);
                                        if (this_atom_dGsolv_tau == 0.0) {
                                            this_atom_dGsolv_tau = approximate_dGsolv_tau;
                                            atomicNumbersWithout_dGsolv_tau.push_back(it.first);
                                        }

                                        E_vdw += this_atom_dGsolv_tau * it.second;
                                    }
                                    double referenceStateCorrection = RT_kcalPerMol * log(molar_volume_ideal_gas / (calculations[calculationIndex].components[i_solvent_component]->molarVolumeAt25C / 1E6));

                                    double E_diel = (calculations[calculationIndex].components[i_component]->epsilonInfinityTotalEnergy - param.dGsolv_E_gas[i_component]) * kcalPerMol_per_Hartree;
                                    double mu_liquid = RT_kcalPerMol * calculations[calculationIndex].lnGammaTotal(i_concentration, i_component);
                                    double E_ring = param.dGsolv_omega_ring * param.dGsolv_numberOfAtomsInRing[i_component];
                                    dGsolv = E_diel + mu_liquid - E_vdw - E_ring - referenceStateCorrection - param.dGsolv_eta;
#ifdef PRINT_DEBUG_INFO
                                    display("dGsolv_1_" + std::to_string(calculationIndex) + ": " + std::to_string(dGsolv) + "\n");
                                    display("E_diel: " + std::to_string(E_diel) + "\n");
                                    display("mu_liquid: " + std::to_string(mu_liquid) + "\n");
                                    display("E_vdw: " + std::to_string(E_vdw) + "\n");
                                    display("E_ring: " + std::to_string(E_ring) + "\n");
                                    display("referenceStateCorrection: " + std::to_string(referenceStateCorrection) + "\n");
                                    display("dGsolv_eta: " + std::to_string(param.dGsolv_eta) + "\n");
#endif

                                }

                                calculations[calculationIndex].dGsolv(i_concentration, i_component) = dGsolv;
                            }
                        }
                    }
                    if (atomicNumbersWithout_dGsolv_tau.size() > 0) {
                        sort(atomicNumbersWithout_dGsolv_tau.begin(), atomicNumbersWithout_dGsolv_tau.end());
                        atomicNumbersWithout_dGsolv_tau.erase(unique(atomicNumbersWithout_dGsolv_tau.begin(), atomicNumbersWithout_dGsolv_tau.end()), atomicNumbersWithout_dGsolv_tau.end());
                        std::string ANs = "";
                        for (int i_AN = 0; i_AN < atomicNumbersWithout_dGsolv_tau.size(); i_AN++) {
                            ANs += std::to_string(atomicNumbersWithout_dGsolv_tau[i_AN]) + ";";
                        }
                        if (param.sw_dGsolv_calculation_strict == 1) {
                            throw std::runtime_error("For the following atomic numbers not all parameters are available for the calculation of solvation energies: " + ANs);
                        }
                        else {
                            warnings.push_back(" - For the following atomic numbers not all parameters are available for the calculation of solvation energies, an estimate was used: " + ANs);
                        }
                    }
                }
            }
            });
    }
    e.rethrow();
}


/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once

#include "helper_functions.hpp"
#include <stdexcept>

// returns lower left triangular matrices, this is because this way the matrix is accessed in sequential order in memory
void calculateInteractionMatrix(segmentTypeCollection& segments, MatrixCalcType& A_int, MatrixCalcType& A_int_PDET, MatrixCalcType& A_int_SecondPDET, std::vector<Eigen::MatrixXd>& partialInteractionMatrices, parameters& param, double temperature) {

    const int numberOfSegments = int(segments.size());
    const int indexOfHole = numberOfSegments - 1;

    double sigmai = 0.0;
    double sigmaj = 0.0;
    double sigmaCorri = 0.0;
    double sigmaCorrj = 0.0;
    double sigmaTransi = 0.0;
    double sigmaTransj = 0.0;
    double sigmaMFij = 0.0;

    double const sigmaHB = param.SigmaHB;
    double const minus_sigmaHB = -1 * sigmaHB;

    std::vector<double> ChargeRaster = param.ChargeRaster;


    // neutral - neutral interactions -------------------------------------------------------------------------------------------------
    // misfit
    double const misfit_prefactor = param.Aeff * param.alpha * 5950000.0 * 0.5;

    // hb
    double CHB_T = 0;
    double CHB_T_PDETemperature = 0;
    double CHB_T_SecondPDETemperature = 0;
    double buffdb1 = 1.0 - param.CHBT + param.CHBT * (298.15 / (temperature));
    if (buffdb1 > 0) { 
        CHB_T = param.CHB * 36700000.0 * buffdb1;
        CHB_T_PDETemperature = -1 * param.CHB * 36700000.0 * param.CHBT * (298.15 / (pow(temperature, 2)));
        CHB_T_SecondPDETemperature = 2 * param.CHB * 36700000.0 * param.CHBT * (298.15 / (pow(temperature, 3)));
    }
    
    double const hb_prefactor = param.Aeff * CHB_T;
    double const hb_prefactor_PDETemperatur = param.Aeff * CHB_T_PDETemperature;
    double const hb_prefactor_SecondPDETemperatur = param.Aeff * CHB_T_SecondPDETemperature;
    
    double val = 0;
    double val2 = 0; // added for partial derivative with respect to temperature
    double val3 = 0; // added for second partial derivative with respect to temperature

    MatrixCalcType binaryComponentDispersionEnergies;
    MatrixCalcType binaryComponentDispersionEnergies_PDETemperature;
    MatrixCalcType binaryComponentDispersionEnergies_SecondPDETemperature;
    
    bool withDerivative = false;

    if (param.sw_phi == 1) {
        int numberOfComponents = param.phi_param.rows();

        // exploit the fact that multiplication is commutative and the matrix is symmetric
        // include hole, interaction energy is zero as expected
        binaryComponentDispersionEnergies = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
        if (A_int_PDET.rows() == numberOfSegments && A_int_PDET.cols() == numberOfSegments) {
            binaryComponentDispersionEnergies_PDETemperature = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
            binaryComponentDispersionEnergies_SecondPDETemperature = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
            withDerivative = true;
        }
        for (int i = 0; i < numberOfComponents; i++) {
            double dispTi = param.phi_param(i, 2) * 100;
            double dispT0i = param.phi_param(i, 1) * 4184;
            double expi = (1 - exp(-dispTi / temperature));
            double dispi = dispT0i * expi;
            for (int j = i; j < numberOfComponents; j++) {
                double dispTj = param.phi_param(j, 2) * 100;
                double dispT0j = param.phi_param(j, 1) * 4184;
                double expj = (1 - exp(-dispTj / temperature));
                double dispj = dispT0j * expj;
                double val = -1 * 0.5 * pow((dispj * dispi), 0.5);
                binaryComponentDispersionEnergies(i, j) = val;
                binaryComponentDispersionEnergies(j, i) = val;
                if (withDerivative) {
                    double num = dispT0i * dispT0j * (dispTi * exp(dispTj / temperature) + dispTj * exp(dispTi / temperature) - dispTj - dispTi) * exp(-dispTj / temperature - dispTi / temperature);
                    double den = 2 * pow(temperature, 2) * pow((dispj * dispi), 0.5);
                    double val2 = 0.5 * num / den;
                    binaryComponentDispersionEnergies_PDETemperature(i, j) = val2;
                    binaryComponentDispersionEnergies_PDETemperature(j, i) = val2;
 
                    double nummod = num / (dispT0i * dispT0j);
                    double denmod = den / sqrt(dispT0i * dispT0j);
                    double bracketForMultiplicator2 = dispTi * exp(dispTj / temperature) - dispTi + dispTj * exp(dispTi / temperature) - dispTj;
                    double dden_dT = (4 * temperature) * sqrt(expi * expj) + pow(temperature, 2) / sqrt(expi * expj) * (-1 * (dispTi / pow(temperature, 2) * exp(-1 * dispTi / temperature)) * expj - expi * (dispTj / pow(temperature, 2) * exp(-1 * dispTj / temperature)));
                    double dnum_dT = exp(-1 * dispTj / temperature - dispTi / temperature) * (-1 * (dispTi * dispTj) / pow(temperature, 2) * (exp(dispTj / temperature) + exp(dispTi / temperature)) + bracketForMultiplicator2 * ((dispTj + dispTi) / pow(temperature, 2)));

                    double val3 = 0.5 * sqrt(dispT0i * dispT0j) * (dnum_dT * denmod - nummod * dden_dT) / pow(denmod, 2);

                    binaryComponentDispersionEnergies_SecondPDETemperature(i, j) = val3;
                    binaryComponentDispersionEnergies_SecondPDETemperature(j, i) = val3;
                }
            }
        }
    }

    int UpperBoundIndexForNeutralComponents = std::max(segments.upperBoundIndexForGroup[0], segments.upperBoundIndexForGroup[1]);
    UpperBoundIndexForNeutralComponents = std::max(UpperBoundIndexForNeutralComponents, segments.upperBoundIndexForGroup[2]);
    UpperBoundIndexForNeutralComponents = std::max(UpperBoundIndexForNeutralComponents, segments.upperBoundIndexForGroup[7]);

    for (int i = segments.lowerBoundIndexForGroup[0]; i < UpperBoundIndexForNeutralComponents; i++) {

        bool i_is_segment_on_water = i >= segments.lowerBoundIndexForGroup[2] && i < segments.upperBoundIndexForGroup[2];
        sigmai = segments.SegmentTypeSigma[i];

        if (param.sw_misfit  > 0) {
            sigmaCorri = segments.SegmentTypeSigmaCorr[i];
            sigmaTransi = sigmaCorri - 0.816 * sigmai;
        }

        for (int j = i; j < UpperBoundIndexForNeutralComponents; j++) {

            val2 = 0;
            val3 = 0;

            sigmaj = segments.SegmentTypeSigma[j];
            sigmaMFij = sigmai + sigmaj;

            if (param.sw_misfit > 0) {
                sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
            }
            else {
                val = misfit_prefactor * sigmaMFij * sigmaMFij;
            }


            if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {

                if (segments.SegmentTypeHBtype[i] == 1 && segments.SegmentTypeHBtype[j] == 2) {

                    val += hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);

                    if (withDerivative) {
                        val2 = hb_prefactor_PDETemperatur * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                        val3 = hb_prefactor_SecondPDETemperatur * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                    }
                }
                else if (segments.SegmentTypeHBtype[i] == 2 && segments.SegmentTypeHBtype[j] == 1) {
                    throw std::runtime_error("This should not happen. Wrong assumption calculating interaction matrix. 1");
                }

            }
            else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                if (segments.SegmentTypeHBtype[i] == 2 && segments.SegmentTypeHBtype[j] == 1) {

                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);

                    if (withDerivative) {
                        val2 = hb_prefactor_PDETemperatur * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                        val3 = hb_prefactor_SecondPDETemperatur * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                    }
                }
                else if (segments.SegmentTypeHBtype[i] == 1 && segments.SegmentTypeHBtype[j] == 2) {
                    throw std::runtime_error("This should not happen. Wrong assumption calculating interaction matrix. 2");
                }

            }

            if (param.sw_phi == 1) {
                int sigmaiMolIndex = segments.SegmentTypeMoleculeIndex[i];
                int sigmajMolIndex = segments.SegmentTypeMoleculeIndex[j];

                val += binaryComponentDispersionEnergies(sigmaiMolIndex, sigmajMolIndex);
                if (withDerivative) {
                    val2 += binaryComponentDispersionEnergies_PDETemperature(sigmaiMolIndex, sigmajMolIndex);
                    val3 += binaryComponentDispersionEnergies_SecondPDETemperature(sigmaiMolIndex, sigmajMolIndex);
                }
            }

            // always j >= i
            A_int(j, i) = static_cast<calcType>(val);
            if (withDerivative) {
                A_int_PDET(j, i) = static_cast<calcType>(val2);
                A_int_SecondPDET(j, i) = static_cast<calcType>(val3);
            }
     	}
    }
    
    if (param.sw_useSegmentReferenceStateForInteractionMatrix == 1) {
        
        // apply the required reference state for the interaction matrix
        for (int i = 0; i < numberOfSegments; i++) {
            for (int j = i + 1; j < numberOfSegments; j++) {
                // always j >= i + 1
                A_int(j, i) = A_int(j, i) - 0.5f*(A_int(i, i) + A_int(j, j));
                if (withDerivative) {
                    A_int_PDET(j, i) = A_int_PDET(j, i) - 0.5f * (A_int_PDET(i, i) + A_int_PDET(j, j));
                    A_int_SecondPDET(j, i) = A_int_SecondPDET(j, i) - 0.5f * (A_int_SecondPDET(i, i) + A_int_SecondPDET(j, j));
                }
            }
        }

        // assign zero to the diagonal
        for (int i = 0; i < numberOfSegments; i++) {
            A_int(i, i) = 0.0f;
            if (withDerivative) {
                A_int_PDET(i, i) = 0.0f;
                A_int_SecondPDET(i, i) = 0.0f;
            }
        }

        if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
            for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                for (int i = 0; i < numberOfSegments; i++) {
                    for (int j = i + 1; j < numberOfSegments; j++) {
                        // always j >= i + 1
                        partialInteractionMatrices[h](j, i) = partialInteractionMatrices[h](j, i) - 0.5 * (partialInteractionMatrices[h](i, i) + partialInteractionMatrices[h](j, j));
                    }
                }
            }

            for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                for (int i = 0; i < numberOfSegments; i++) {
                    partialInteractionMatrices[h](i, i) = 0.0;
                }
            }
        }
    }
}

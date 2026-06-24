/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once
#define NOMINMAX
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <unordered_map>
#include <numeric>
#include <memory>
#include <algorithm>
#include <mutex>


// Constants
#define PI 3.14159265358979323846
#define R_GAS_CONSTANT 8.31446261815324
#define N_AVOGADRO 6.022140857e+23
#define MAX_CONCENTRATION_DIFF_FROM_ZERO 0.000001
#define VMOL_V 6.022140857e-7

/* Structures and classes */

template <typename T>
void apply_vector_permutation_in_place(std::vector<T>& vec, const std::vector<int>& p) {
	std::vector<bool> done(vec.size());
	for (int i = 0; i < vec.size(); ++i) {
		if (done[i])
		{
			continue;
		}
		done[i] = true;
		int prev_j = i;
		int j = p[i];
		while (i != j) {
			std::swap(vec[prev_j], vec[j]);
			done[j] = true;
			prev_j = j;
			j = p[j];
		}
	}
}

struct parameters {

	parameters() : phi_param(NULL, 0, 0) {	}

    /* General switches */
	int sw_phi = 0;					// Whether or not to uss the EOS
	int sw_PT_Flash = 0;
	int sw_checkAndCalcBinaryLLE = 0;
	int sw_lnGamma = 0;
	int sw_isotherm = 0;
	int sw_excess = 0;
	int sw_Pbub = 0;
	int sw_Pdew = 0;
	int sw_Tbub = 0;
	int sw_Tdew = 0;
	int sw_criticalPoint = 0;
	int sw_criticalPoint_specifiedx = 0;
	int sw_azeotropicPoint = 0;
    int sw_misfit = 2;              /* switch: "0" to use contribution without accounting for sigma correlation. 
                                           "1" to account for sigma correlation in misfit contribution 
										   "2" to account for sigma correlation in misfit contribution except for ions */

	int sw_useSegmentReferenceStateForInteractionMatrix = 0;		/* switch: "0" Set segment reference state to pure segment (COSMO-RS default) (segment reference state cancels if molecular refstate calculated)
																	   "1" Set segment reference state to conductor (segment reference state cancels if molecular refstate calculated) */

   	int sw_combTerm = 1;            /* switch:  "0" No combinatorial term
											"1" to use the combinatorial term by Staverman-Guggenheim
											"2" to use the combinatorial term by Klamt (2003)
											"3" to use modified Staverman-Guggenheim combinatorial term with exponential scaling */

	int sw_atomicNumber = 1; // whether or not to use the atomic number as descriptor

	int sw_differentiateHydrogens = 0;
	int sw_differentiateMoleculeGroups = 0;

	std::string sw_COSMOfiles_type = "ORCA_COSMO_TZVPD"; // Type of COSMOfile used, this is used to know which function to use to load the sigma profile. e.g. Turbomole

	int sw_calculateContactStatisticsAndAdditionalProperties = 0;		/* switch:  "0" Do not calculate contact statistics
																				"1" Calculate contact statistics 
																				"2" Calculate contact statistics, partial molar properties and average surface energies*/
	int numberOfPartialInteractionMatrices;

	int sw_alwaysReloadSigmaProfiles = 0;

	int sw_reloadConcentrations = 0;
	int sw_reloadReferenceConcentrations = 0;

	int sw_alwaysCalculateSizeRelatedParameters = 0;	/* switch: "0" combinatorial term and segment fraction is only calculated once at first execution
															   "1" combinatorial term and segment fraction  is calculated on every iteration */

	int sw_skip_COSMOSPACE_errors = 0;	/* switch: "0" if COSMOSPACE does not converge, it stops execution showing DEBUG information
											   "1" if COSMOSPACE does not converge, execution continues setting the objective function very high */

	int sw_dGsolv_calculation_strict = 1; // 0Allows calculation of solvation free energies also for atoms that have not been parameterized, but gives a warning
										  // 1: Allows calculation of solvation free energies if all parameters are available

	int sw_calculateResidualProperties = 0;		    // 0: Do not calculate partial derivative of Segment Gammas with respect to temperature and molar volume
																    // 1: calculate partial derivative of Segment Gammas with respect to temperature and molar volume
																	//    ONLY WORKS WHEN SW_Phi = 1

    /* COSMO-RS MODEL PARAMETERS */
    /* General parameters */
    double Aeff;                    /* area of effective thermodynamic contact  [Angstrom^2] */
    double alpha;                   /* misfit energy prefactor (alpha')  [J*Angstrom^2/(mol*e^2)] */
    double CHB;                     /* hydrogen bond prefactor [J*Angstrom^2/(mol*e^2)] */
    double CHBT;                    /* hydrogen bond temperature parameter */
    double SigmaHB;                 /* sigma threshhold for hydrogen bond [e/Angstrom^2] */
    double Rav;                     /* Averaging Radius for sigma averaging [Angstrom] */
    double RavCorr;                 /* Averaging Radius to determine sigma correlation [Angstrom]*/
    double fCorr;                   /* Correction factor for the introduction of sigma correlation to the misfit energy */
	double comb_SG_A_std;           /* Standard area for Staverman-Guggenheim combinatorial contribution [Angstrom^2] */
	double comb_SG_z_coord;         /* Coordination number for Staverman Guggenheim contribution */
	double comb_modSG_exp;			/* Exponent parameter for modified Staverman-Guggenheim contribution with exponential scaling [-] */
	double comb_SGG_lambda;         /* Parameter for the mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */
	double comb_SGG_beta;         	/* Parameter for the mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */

	double comb_lambda0;			/* lambda paramters used in the calculation of the combinatorial contribuation from Klamt or Frank & Hannebauer */
	double comb_lambda1;
	double comb_lambda2;

	double hole_area;

	/* Parameters used for solvation energy calculation */
	double dGsolv_eta;
	double dGsolv_omega_ring;
	std::vector<double> dGsolv_tau = std::vector<double>(118, 0.0);
	std::vector<double> dGsolv_E_gas;
	std::vector<int> dGsolv_numberOfAtomsInRing;
	std::vector<double> dGsolv_molarVolume;

    /* Radii used for cosmo segment scaling for monoatomic ions */
	std::vector<double> R_i = std::vector<double>(118, 0.0);                       /* new radii, element specific (r_i[AN]) from the input file*/
	std::vector<double> R_i_COSMO = std::vector<double>(118, 0.0);                 /* old radii from COSMO-file, element specific (r_i[AN]); */

    /* Experimental parameters */
    /* These are used to test new modifications to COSMO-RS. Parameters of succesfull modifications should
    *  later be implemented with standard parameters */
	std::unordered_map<std::string,double> exp_param;

	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> phi_param;


	// the following vectors may contain hyperparameters / external values for the calculation of mixture properties
	bool useGivenInitialK;
	bool noTracing;
	std::vector<double> temperatures_mixing;
	std::vector<double> pressures_mixing;
	std::vector<std::vector<double>> concentration_x_phase_mixing;
	std::vector<std::vector<double>> concentration_y_phase_mixing;


	std::vector<int> HBClassElmnt = std::vector<int>(300, 0);

	double sigmaMin = -0.15;
	double sigmaMax = 0.15;
	double sigmaStep = 0.001;

	std::vector<double> ChargeRaster;

};

struct segmentTypeCollection {

private:
	std::vector<double> SegmentTypeAreasRowTemplate;

	std::vector<int> get_permutation_vector()
	{
		std::vector<int> p(SegmentTypeGroup.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](int i, int j) {

			if (SegmentTypeMoleculeIndex[i] != SegmentTypeMoleculeIndex[j]) { return SegmentTypeMoleculeIndex[i] < SegmentTypeMoleculeIndex[j]; }

			if (SegmentTypeGroup[i] != SegmentTypeGroup[j]) { return SegmentTypeGroup[i] < SegmentTypeGroup[j]; }

			// if both segments belong to an monoatomic ion, first sort by atomic number
			if (SegmentTypeGroup[i] == 3 || SegmentTypeGroup[i] == 5) {
				if (SegmentTypeAtomicNumber[i] != SegmentTypeAtomicNumber[j]) { return SegmentTypeAtomicNumber[i] < SegmentTypeAtomicNumber[j]; }
			}

			if (SegmentTypeSigma[i] != SegmentTypeSigma[j]) { return SegmentTypeSigma[i] < SegmentTypeSigma[j]; }

			if (SegmentTypeSigmaCorr[i] != SegmentTypeSigmaCorr[j]) { return SegmentTypeSigmaCorr[i] < SegmentTypeSigmaCorr[j]; }

			if (SegmentTypeHBtype[i] != SegmentTypeHBtype[j]) { return SegmentTypeHBtype[i] < SegmentTypeHBtype[j]; }

			if (SegmentTypeAtomicNumber[i] != SegmentTypeAtomicNumber[j]) { return SegmentTypeAtomicNumber[i] < SegmentTypeAtomicNumber[j]; }

			return i < j;

		});
		return p;
	}


public:

	int lowerBoundIndexForGroup[8] = { 0 };
	int upperBoundIndexForGroup[8] = { 0 };
	int numberOfSegmentsForGroup[8] = { 0 };

	segmentTypeCollection() {
		segmentTypeCollection(1);
	}

	segmentTypeCollection(int numberOfMolecules) {
		
		for (int i = 0; i < numberOfMolecules; i++) {
			SegmentTypeAreasRowTemplate.push_back(0.0);
		}
		SegmentTypeAreasRowTemplate.shrink_to_fit();
	}

	std::vector<std::vector<double>> SegmentTypeAreas;
	std::vector<unsigned short> SegmentTypeGroup;
	std::vector<double> SegmentTypeSigma;
	std::vector<double> SegmentTypeSigmaCorr;
	std::vector<unsigned short> SegmentTypeHBtype;
	std::vector<unsigned short> SegmentTypeAtomicNumber;
	std::vector<int> SegmentTypeMoleculeIndex;

	void clear() {

		SegmentTypeAreas.clear();
		SegmentTypeGroup.clear();
		SegmentTypeSigma.clear();
		SegmentTypeSigmaCorr.clear();
		SegmentTypeHBtype.clear();
		SegmentTypeAtomicNumber.clear();
		SegmentTypeMoleculeIndex.clear();
	}

	void reserve(int numberOfSegmentsTypes) {

		SegmentTypeAreas.reserve(numberOfSegmentsTypes);
		SegmentTypeGroup.reserve(numberOfSegmentsTypes);
		SegmentTypeSigma.reserve(numberOfSegmentsTypes);
		SegmentTypeSigmaCorr.reserve(numberOfSegmentsTypes);
		SegmentTypeHBtype.reserve(numberOfSegmentsTypes);
		SegmentTypeAtomicNumber.reserve(numberOfSegmentsTypes);
		SegmentTypeMoleculeIndex.reserve(numberOfSegmentsTypes);
	}

	void shrink_to_fit() {

		for (int i = 0; i < SegmentTypeAreas.size(); i++) {
			SegmentTypeAreas[i].shrink_to_fit();
		}

		SegmentTypeAreas.shrink_to_fit();
		SegmentTypeSigma.shrink_to_fit();
		SegmentTypeSigmaCorr.shrink_to_fit();
		SegmentTypeHBtype.shrink_to_fit();
		SegmentTypeAtomicNumber.shrink_to_fit();
		SegmentTypeMoleculeIndex.shrink_to_fit();
	}

	size_t size() {
		return SegmentTypeHBtype.size();
	}

	void add(unsigned short ind_molecule, unsigned short group, double Sigma, double SigmaCorr, unsigned short HBtype, unsigned short atomicNumber, double Area, int moleculeIndex) {

		if (Area == 0) {
			return;
		}

		int index = -1;
		for (int i = 0; i < SegmentTypeHBtype.size(); i++) {
			if (SegmentTypeGroup[i] == group &&
				SegmentTypeHBtype[i] == HBtype &&
				SegmentTypeSigma[i] == Sigma &&
				SegmentTypeSigmaCorr[i] == SigmaCorr &&
				SegmentTypeAtomicNumber[i] == atomicNumber &&
				SegmentTypeMoleculeIndex[i] == moleculeIndex)  {
				index = i;
				break;
			}
		}

		if (index == -1) {
			SegmentTypeGroup.push_back(group);
			SegmentTypeHBtype.push_back(HBtype);
			SegmentTypeSigma.push_back(Sigma);
			SegmentTypeSigmaCorr.push_back(SigmaCorr);
			SegmentTypeAtomicNumber.push_back(atomicNumber);
			SegmentTypeMoleculeIndex.push_back(moleculeIndex);

			index = (int)SegmentTypeHBtype.size() - 1;
			SegmentTypeAreas.push_back(SegmentTypeAreasRowTemplate);
		}

		SegmentTypeAreas[index][ind_molecule] += Area;
	}

	void sort() {
		std::vector<int> p = get_permutation_vector();

		apply_vector_permutation_in_place(SegmentTypeGroup, p);
		apply_vector_permutation_in_place(SegmentTypeSigma, p);
		apply_vector_permutation_in_place(SegmentTypeSigmaCorr, p);
		apply_vector_permutation_in_place(SegmentTypeHBtype, p);
		apply_vector_permutation_in_place(SegmentTypeAtomicNumber, p);
		apply_vector_permutation_in_place(SegmentTypeAreas, p);
		apply_vector_permutation_in_place(SegmentTypeMoleculeIndex, p);

		int temporaryindex = -1;
		int i;
		for (i = 0; i < size(); i++) {

			if (temporaryindex != SegmentTypeGroup[i]) {

				if (temporaryindex != -1) {
					upperBoundIndexForGroup[temporaryindex] = i;
				}
				temporaryindex = SegmentTypeGroup[i];
				lowerBoundIndexForGroup[temporaryindex] = i;
			}

		}
		upperBoundIndexForGroup[temporaryindex] = i;

		for (i = 0; i < 7; i++) {
			numberOfSegmentsForGroup[i] = std::max(upperBoundIndexForGroup[i] - lowerBoundIndexForGroup[i], 0);
		}
	}
};

struct molecule {
	/* segment properties */
	segmentTypeCollection segments;

	molecule() {
		segments = segmentTypeCollection(1);
	}

	std::string name;
	std::string qmMethod;

	int index;
	double Area;
	double Volume;
	double epsilonInfinityTotalEnergy;
	double molarVolumeAt25C;

	//double critical_T;
	//double critical_P;

	signed char moleculeCharge;
	unsigned short moleculeGroup;

	// Possible groups
	/*
	0 monoatomic neutral
	1 polyatomic neutral
	2 water
	3 monoatomic cation
	4 polyatomic cation
	5 monoatomic anion
	6 polyatomic anion
	*/

	// Atoms
	Eigen::MatrixXd atomPositions;
	Eigen::VectorXd atomRadii;
	Eigen::VectorXi atomAtomicNumbers;

	// Segment information directly from the input file
	Eigen::MatrixXd segmentPositions;
	Eigen::VectorXi segmentAtomIndices;
	Eigen::VectorXi segmentAtomicNumber;
	Eigen::VectorXi segmentHydrogenBondingType;
	Eigen::VectorXd segmentAreas;
	Eigen::VectorXd segmentSigmas;

	void clear_unneeded_matrices(bool keepDataNeededForReloadingSigmaProfile = false) {
		atomPositions.resize(0, 0);
		atomRadii.resize(0);
		// atomAtomicNumbers.resize(0); this is currently needed for monoatomic ions

		segmentAtomIndices.resize(0);

		if (keepDataNeededForReloadingSigmaProfile == false) {
			segmentAtomicNumber.resize(0);
			segmentPositions.resize(0, 0);
			segmentHydrogenBondingType.resize(0);
			segmentAreas.resize(0);
			segmentSigmas.resize(0);
		}

	}
};

struct calculation {

	int number = -1;
	std::string calculationType = "";

	segmentTypeCollection segments;

	calculation(int numberOfMolecules) : \
		contactStatistics(NULL, 0, 0, 0),\
		averageSurfaceEnergies(NULL, 0, 0, 0, 0), \
		partialMolarEnergies(NULL, 0, 0, 0), \
		lnGammaCombinatorial(NULL, 0,0), \
		lnGammaResidual(NULL, 0, 0), \
		lnGammaTotal(NULL, 0, 0), \
		lnPhiRepulsive(NULL, 0, 0), \
		lnPhiAttractive(NULL, 0, 0), \
		lnPhiTotal(NULL, 0, 0), \
		originalConcentrations(NULL, 0, 0), \
		calculatedPartitionCoefficients(NULL, 0, 0), \
		molarVolume(NULL, 0), \
		repulsivePressure(NULL, 0), \
		attractivePressure(NULL, 0), \
		totalPressure(NULL, 0), \
		targetPressure(NULL, 0), \
		molarResidualEnthalpy(NULL, 0), \
		molarResidualHeatCapacityAtConstantPressure(NULL, 0), \
		hE(NULL, 0), \
		gE(NULL, 0), \
		cpE(NULL, 0), \
		hM(NULL, 0), \
		gM(NULL, 0), \
		cpM(NULL, 0), \
		targetTemperature(NULL, 0), \
		criticalT(NULL, 0), \
		criticalP(NULL, 0), \
		criticalx(NULL, 0, 0), \
		dGsolv(NULL, 0, 0)
	
	{
		segments = segmentTypeCollection(numberOfMolecules);
		moleculeIndices = std::vector<int>(numberOfMolecules);
	}

	std::vector<std::shared_ptr<molecule>> components;
	std::vector<int> moleculeIndices;


	std::vector<std::vector<double>> concentrations;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> originalConcentrations;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> calculatedPartitionCoefficients;
	MatrixCalcType segmentConcentrations;
	MatrixCalcType PartialsegmentConcentrationsPartialv;
	MatrixCalcType segmentGammas;
	MatrixCalcType segmentGammasPDETemperature;
	MatrixCalcType segmentGammasPDEVolume;
	MatrixCalcType segmentGammasSecondPDETemperature;
	std::vector<std::vector<int>> referenceStateCalculationIndices;
	std::vector<unsigned short> referenceStateType;
	
	Eigen::MatrixXd PhiDash_pxi;
	Eigen::MatrixXd ThetaDash_pxi;

	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaCombinatorial;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaResidual;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaTotal;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiRepulsive;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiAttractive;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiTotal;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> gE;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> hE;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> cpE;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> gM;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> hM;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> cpM;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> targetTemperature;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> criticalT;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> criticalP;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> criticalx;
	
	MatrixCalcType lnPhiTotalPDETAtConstantPx;
	MatrixCalcType lnGammaResidualPDETatConstantvx;
	MatrixCalcType lnGammaResidualSecondPDETatConstantvx;
	MatrixCalcType lnGammaResidualPDEvatConstantTx;
	bool recalculateConcentrationsForPhi = false;

	Eigen::VectorXd componentCavityVolume;
	Eigen::VectorXd componentHoleVolume;

	Eigen::VectorXd mixtureCavityVolume;
	Eigen::VectorXd mixtureHoleVolume;

	Eigen::MatrixXd PartitionCoefficients;
	Eigen::MatrixXi IndicesForPureComponentCalculations;
	std::vector<double> uniqueTemperatures;
	std::vector<double> uniquePressures;
	Eigen::MatrixXd SaturationPressures;
	Eigen::MatrixXd SaturationTemperatures;
	Eigen::MatrixXd overallConcentrations;
	int helperValue;

	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> molarVolume;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> repulsivePressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> attractivePressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> targetPressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> totalPressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> molarResidualEnthalpy;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> molarResidualHeatCapacityAtConstantPressure;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dGsolv;

	Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> contactStatistics;
	Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>> averageSurfaceEnergies;
	Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> partialMolarEnergies;

	
	// the following matrices are to save the computations temporarily if needed
	// for internal use. Access is through the respective maps above.
	Eigen::MatrixXd concentrations_data;
	Eigen::MatrixXd lnGammaCombinatorial_data;
	Eigen::MatrixXd lnGammaResidual_data;
	Eigen::MatrixXd lnGammaTotal_data;
	Eigen::MatrixXd dGsolv_data;
	Eigen::VectorXd molarVolume_data;
	Eigen::VectorXd repulsivePressure_data;
	Eigen::VectorXd attractivePressure_data;
	Eigen::VectorXd totalPressure_data;
	Eigen::VectorXd targetPressure_data;
	Eigen::VectorXd targetTemperature_data;
	Eigen::MatrixXd lnPhiTotal_data;
	Eigen::MatrixXd lnPhiRepulsive_data;
	Eigen::MatrixXd lnPhiAttractive_data;
	Eigen::VectorXd criticalP_data;
	Eigen::VectorXd criticalT_data;
	Eigen::MatrixXd criticalx_data;
	Eigen::VectorXd molarResidualEnthalpy_data;
	Eigen::VectorXd molarResidualHeatCapacityAtConstantPressure_data;

	Eigen::Tensor<double, 3, Eigen::RowMajor> contactStatistics_data;
	Eigen::Tensor<double, 4, Eigen::RowMajor> averageSurfaceEnergies_data;
	Eigen::Tensor<double, 3, Eigen::RowMajor> partialMolarEnergies_data;
	
	//holds first and last segment index that are needed for calculations with the COSMOSPACE equation allowing to save calculations
	std::vector<int> lowerBoundIndexForCOSMOSPACECalculation; 
	std::vector<int> upperBoundIndexForCOSMOSPACECalculation;

	std::vector<double> temperatures;
	std::vector<double> pressures;

	std::vector<std::vector<int>> TauConcentrationIndices;
	std::vector<double> TauTemperatures;
	std::vector<MatrixCalcType>Taus;

	std::vector<int> actualConcentrationIndices;

	size_t originalNumberOfCalculations;

	void invalidateTaus() {

		if (Taus.size() != 0){
			for (int i = 0; i < TauTemperatures.size(); i++) {
				Taus[i] = MatrixCalcType::Zero(0, 0);
			}
		}
	}

	int addOrFindTauIndexForConditions(double temperature) {

		int index = -1;

		for (int i = 0; i < TauTemperatures.size(); i++) {
			if (TauTemperatures[i] == temperature) {

				index = i;
				break;
			}
		}

		if (index == -1) {
			TauTemperatures.push_back(temperature);
			Taus.push_back(MatrixCalcType::Zero(0, 0));
			TauConcentrationIndices.push_back(std::vector<int>());

			index = int(TauTemperatures.size()) - 1;
		}

		return index;
	}

	int addOrFindArrayIndexForConcentration(std::vector<double> concentration, double temperature, double pressure = -1)
	{
		int index = -1;
		int nConditions = 1; // only temperature
		for (int i = 0; i < concentrations.size(); i++) {
			int nEqual = 0;
			for (int j = 0; j < concentration.size(); j++) {
				if (concentrations[i][j] == concentration[j]) {
					nEqual++;
				}
			}

			// temperature
			if (temperatures[i] == temperature) {
				nEqual++;
			}

			// if concentrations and conditions the same, we found the guy
			if (nEqual == concentration.size() + nConditions) {
				index = i;
				break;
			}
		}

		if (index == -1) {
			concentrations.push_back(std::move(concentration));
			temperatures.push_back(temperature);

			// When calculating excess propoerties with openCOSMO-RS-Phi also pressure must be considered as fugacity coefficient = f(T,P,x)
			if (pressure != -1) {
				pressures.push_back(pressure);
			}

			index = int(concentrations.size()) - 1;
		}

		return index;
	}

	int addOrFindArrayIndexInSubcalculations(std::vector<calculation> subcalculations, std::vector<std::shared_ptr<molecule>> molecules, double temperature = -1, double pressure = -1)
	{
		int index = -1;
		int nConditions = 0;
		if (temperature != -1) {
			nConditions += 1;
		}
		if (pressure != -1) {
			nConditions += 1;
		}
		for (int i = 0; i < subcalculations.size(); i++) {
			int nEqual = 0;
			for (int j = 0; j < molecules.size(); j++) {
				if (subcalculations[i].components[j] == molecules[j]) {
					nEqual++;
				}
			}

			// temperature
			if (temperature != -1) {
				if (subcalculations[i].temperatures[0] == temperature) {
					nEqual++;
				}
			}

			// pressure if also part of checked conditions
			if (pressure != -1) {
				if (subcalculations[i].targetPressure(0) == pressure) {
					nEqual++;
				}
			}

			// if concentrations and conditions the same, we found the guy
			if (nEqual == molecules.size() + nConditions) {
				index = i;
				break;
			}
		}

		return index;
	}


	void shrink_to_fit() {

		for (int i = 0; i < concentrations.size(); i++) {
			concentrations[i].shrink_to_fit();
		}

		for (int i = 0; i < TauConcentrationIndices.size(); i++) {
			TauConcentrationIndices[i].shrink_to_fit();
		}

		TauTemperatures.shrink_to_fit();

		actualConcentrationIndices.shrink_to_fit();
		TauConcentrationIndices.shrink_to_fit();

		concentrations.shrink_to_fit();

		for (int i = 0; i < referenceStateCalculationIndices.size(); i++) {
			referenceStateCalculationIndices[i].shrink_to_fit();
		}
		referenceStateCalculationIndices.shrink_to_fit();
		referenceStateType.shrink_to_fit();

		lowerBoundIndexForCOSMOSPACECalculation.shrink_to_fit();
		upperBoundIndexForCOSMOSPACECalculation.shrink_to_fit();

		temperatures.shrink_to_fit();
	}
};

/* this class is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section */
class threadException {
	std::exception_ptr Ptr;
	std::mutex         Lock;
public:
	threadException() : Ptr(nullptr) {}

	void rethrow() {
		if (this->Ptr) std::rethrow_exception(this->Ptr);
	}
	void capture_exception() {
		std::unique_lock<std::mutex> guard(this->Lock);
		this->Ptr = std::current_exception();
	}

	template <typename Function, typename... Parameters>
	void run(Function f, Parameters... params)
	{
		if (!this->Ptr) {
			try
			{
				f(params...);
			}
			catch (...)
			{
				capture_exception();
			}
		}
	}
};

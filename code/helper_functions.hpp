/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once
#include "general.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

std::function<void(std::string)> display;
std::function<void(std::string, unsigned long)> displayTime;

static inline std::string leftTrim(std::string val) {
	val.erase(val.begin(), std::find_if(val.begin(), val.end(), [](unsigned char c) {
		return !std::isspace(c);
		}));

	return val;
}

static inline std::string rightTrim(std::string val) {
	val.erase(std::find_if(val.rbegin(), val.rend(), [](unsigned char c) {
		return !std::isspace(c);
		}).base(), val.end());
	return val;
}

static inline std::string trim(std::string val) {
	val = rightTrim(val);
	val = leftTrim(val);
	return val;
}

static inline std::string replace(std::string haystack, const std::string& from, const std::string& to) {
	size_t start_pos = haystack.find(from);
	if (start_pos == std::string::npos)
		return haystack;
	haystack.replace(start_pos, from.length(), to);
	return haystack;
}

static inline bool startsWith(std::string& haystack, const std::string& needle) {
	return haystack.rfind(needle, 0) == 0;
}

static inline bool endsWith(std::string& haystack, const std::string& needle) {

	if (needle.size() > haystack.size()) return false;

	return std::equal(needle.rbegin(), needle.rend(), haystack.rbegin());
}

static inline std::vector<std::string> split(const std::string& haystack, char delimiter)
{
	std::vector<std::string> retVal;
	std::stringstream _stringstream(haystack);
	std::string token;
	while (std::getline(_stringstream, token, delimiter)) {
		retVal.push_back(token);
	}
	return retVal;
}
static inline double round_and_truncate(double number_val, int n) {
	number_val = round(number_val * pow(10, n)) / pow(10, n);

	double factor = 1;
	double previous = std::trunc(number_val); // remove integer portion
	number_val -= previous;
	for (int i = 0; i < n; i++) {
		number_val *= 10;
		factor *= 10;
	}
	number_val = std::trunc(number_val);
	number_val /= factor;
	number_val += previous; // add back integer portion
	return number_val;
}

// Returns next greater multiple of 8
// if x is divisible by 8 it returns x
static inline int RoundUpToNextMultipleOfEight(int x) {
	return ((x + 7) & (-8));
}

// Returns previous multiple of 8
// if x is divisible by 8 it returns x
static inline int RoundDownToNextMultipleOfEight(int x) {
	if (x % 8 == 0) {
		return x;
	}

	return RoundUpToNextMultipleOfEight(x) - 8;
}

#include <iostream>
#include <cmath>

// from pseudocode https://en.wikipedia.org/wiki/ITP_method
template <typename Func>
double ITP(Func f, double a, double b, double y_a = std::nan("1"), double y_b = std::nan("1"), double x_atol = std::numeric_limits<double>::epsilon(), double y_atol = std::numeric_limits<double>::epsilon(), double kappa1 = std::nan("1"), double kappa2 = 2, double n0 = 1, bool ErrorMessage = true) {
	
	if (std::isnan(y_a))
		y_a = f(a);

	if (std::isnan(y_b))
		y_b = f(b);

	if (y_a * y_b > 0 && ErrorMessage == true)
		throw std::runtime_error("The function evaluated at the given interval does not return values with opposite sign.");
	else if (y_a * y_b > 0 && ErrorMessage == false)
		return std::nan("1");
	
	// make b larger than a
	if (a > b) {
		std::swap(a, b);
		std::swap(y_a, y_b);
	}
	
	// Validate hyperparameters
	if (std::isnan(kappa1))
		kappa1 = 0.2 / abs(b - a);
	if (!(0 < kappa1 && kappa1 < std::numeric_limits<double>::infinity()))
		throw std::runtime_error("Invalid value for kappa1");

	if (!(1 <= kappa2 && kappa2 < 1 + (1 + sqrt(5)) / 2))
		throw std::runtime_error("Invalid value for kappa2");

	if (!(0 < n0 && n0 < std::numeric_limits<double>::infinity()))
		throw std::runtime_error("Invalid value for n0");

	int n_1div2 = ceil(log2((b - a) / (2 * x_atol)));
	int n_max = n_1div2 + static_cast<int>(std::ceil(n0));

	// the original pseudocode assumes y_a < 0 < y_b:
	double multiplier = 1.0;
	if (y_a > y_b)
		multiplier = -1.0;

	y_a = y_a * multiplier;
	y_b = y_b * multiplier;

	int j = 0;
	while (b - a > 2 * x_atol && abs(y_b - y_a) > y_atol && j < n_max) {

		// calculate parameters
		double x_1div2 = (a + b) / 2;
		double r = x_atol * pow(2.0, n_max - j) - (b - a) / 2;
		if (r < 0.0)
			r = 0.0;
		double delta = kappa1 * pow((b - a), kappa2);

		// interpolate
		double x_f = (y_b * a - y_a * b) / (y_b - y_a);

		// truncate
		double sigma = (x_1div2 - x_f) > 0 ? 1 : -1;
		double x_t = (delta <= abs(x_1div2 - x_f)) ? (x_f + sigma * delta) : x_1div2;

		// project
		double x_ITP = (abs(x_t - x_1div2) <= r) ? x_t : (x_1div2 - sigma * r);

		// update
		double y_ITP = multiplier * f(x_ITP);
		if (y_ITP > 0) {
			b = x_ITP;
			y_b = y_ITP;
		}
		else if (y_ITP < 0) {
			a = x_ITP;
			y_a = y_ITP;
		}
		else {
			a = b = x_ITP;
		}

		j += 1;
	}

	return (a + b) / 2;
}

template <typename Func>
double bisection(Func f, double a, double b, double y_a = std::nan("1"), double y_b = std::nan("1"), double y_atol = std::numeric_limits<double>::epsilon()) {
	
	if (std::isnan(y_a))
		y_a = f(a);

	if (std::isnan(y_b))
		y_b = f(b);

	if (y_a * y_b > 0)
		throw std::runtime_error("The function evaluated at the given interval does not return values with opposite sign.");

	const double scale = std::max(std::abs(a), std::abs(b));
	const double x_atol = std::max(scale * 1e-12, 1e-15);

	double y_tol = y_atol;
	if (y_tol <= std::numeric_limits<double>::epsilon())
		y_tol = 1e-12;

	double c = a;
	double y_c = y_a;

	int n_max = 0;
	if (b > a) {
		double n_est = std::log2((b - a) / x_atol);
		n_max = (n_est > 0.0) ? static_cast<int>(std::ceil(n_est)) + 2 : 64;
	}
	else {
		n_max = 64;
	}

	int iter = 0;
	while ((b - a) > x_atol && iter < n_max) {
		c = 0.5 * (a + b);
		y_c = f(c);

		if (std::abs(y_c) <= y_tol)
			break;

		if (y_a * y_c < 0.0) {
			b = c;
			y_b = y_c;
		}
		else {
			a = c;
			y_a = y_c;
		}

		++iter;
	}

	return 0.5 * (a + b);
}
template<typename T>
std::string convertPointerToString(T* pointer) {
	std::ostringstream oss;
	oss << pointer;
	return oss.str();
}

template <typename T>
void Write1DArrayToFile(std::string path, const T* array_ptr, int n_rows, int n_cols,
	bool transpose = false, int print_to_row_n = -1, int print_to_col_n = -1,
	int n_repeat = 1) {

	// Adjust print_to_row_n and print_to_col_n if they are not provided
	if (print_to_row_n == -1) {
		print_to_row_n = n_rows;
	}
	if (print_to_col_n == -1) {
		print_to_col_n = n_cols;
	}

	// Open file using RAII (std::ofstream)
	std::ofstream file(path.data());
	if (!file) {
		throw std::runtime_error("Could not open file: " + std::string(path));
	}

	// Set formatting options: fixed point, 6 decimal precision
	file << std::fixed << std::setprecision(12);

	// Loop through the data, handling transposition if needed
	if (transpose) {
		for (int h = 0; h < n_repeat; ++h) {
			for (int i = 0; i < print_to_col_n; ++i) {
				for (int j = 0; j < print_to_row_n; ++j) {
					file << std::setw(14) << array_ptr[j * n_cols + i] << "  ";
				}
				file << "\n";
			}
		}
	}
	else {
		for (int h = 0; h < n_repeat; ++h) {
			for (int i = 0; i < print_to_row_n; ++i) {
				for (int j = 0; j < print_to_col_n; ++j) {
					file << std::setw(14) << array_ptr[i * n_cols + j] << "  ";
				}
				file << "\n";
			}
		}
	}

	// Check if any errors occurred during writing
	if (!file) {
		throw std::runtime_error("Error occurred while writing to file: " + std::string(path));
	}
}

void WriteEigenMatrixToFile(std::string path, const Eigen::MatrixXd& m, std::string headline = "") {
	std::ofstream file(path.data());
	if (!file) {
		throw std::runtime_error("Could not open file: " + std::string(path));
	}

	file.precision(8);  // Set precision to 6 digits
	file << std::scientific;  // Use scientific notation

	if (headline != "") {
		file << headline << " ";
		file << "\n";
	}

	for (int i = 0; i < m.rows(); ++i) {
		for (int j = 0; j < m.cols(); ++j) {
			file << m(i, j) << " ";
		}
		file << "\n";
	}

	if (!file) {
		throw std::runtime_error("Error occurred while writing to file: " + std::string(path));
	}
}

void WriteEigenVectorToFile(std::string path, const Eigen::VectorXd& m, std::string headline = "") {
	std::ofstream file(path.data());
	if (!file) {
		throw std::runtime_error("Could not open file: " + std::string(path));
	}

	file.precision(15);  // Set precision to 15 digits
	file << std::scientific;  // Use scientific notation

	if (headline != "") {
		file << headline << " ";
		file << "\n";
	}

	for (int i = 0; i < m.size(); ++i) {
		file << m(i) << " ";
		file << "\n";
	}

	if (!file) {
		throw std::runtime_error("Error occurred while writing to file: " + std::string(path));
	}
}

void WriteExtendedSigmaProfileToFile(std::string path, segmentTypeCollection& segments) {

	std::ofstream file(path.data());
	if (!file) {
		throw std::runtime_error("Could not open file: " + std::string(path));
	}

	file << std::fixed << std::setprecision(6);

	// Write segment data to file
	for (int i = 0; i < segments.size(); ++i) {
		file << std::setw(4) << i << "  "
			<< std::setw(14) << segments.SegmentTypeSigma[i] << "  "
			<< std::setw(14) << segments.SegmentTypeSigmaCorr[i] << "  "
			<< std::setw(4) << segments.SegmentTypeAtomicNumber[i] << "  "
			<< std::setw(3) << segments.SegmentTypeHBtype[i] << "  "
			<< std::setw(3) << segments.SegmentTypeGroup[i] << "  "
			<< std::setw(14) << segments.SegmentTypeAreas[i][0] << "\n";
	}

	if (!file) {
		throw std::runtime_error("Error occurred while writing to file: " + std::string(path));
	}
}

// Funktion zur Berechnung des Schnittpunkts zweier Liniensegmente
bool findSegmentIntersection(
	double x1, double y1, double x2, double y2,
	double x3, double y3, double x4, double y4,
	std::pair<double, double>& intersection) {

	double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
	if (std::abs(denom) < 1e-10) return false; // Segmente sind parallel

	double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
	double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

	// Ausschluss von Endpunkten durch strenge Ungleichungen
	if (t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0) {
		intersection.first = x1 + t * (x2 - x1);
		intersection.second = y1 + t * (y2 - y1);
		return true;
	}
	return false;
}

// Funktion zur linearen Interpolation von x basierend auf a2-Werten
double interpolateX(double x1, double x2, double a_val1, double a_val2, double a_target) {
	if (std::abs(a_val2 - a_val1) < 1e-10) return (x1 + x2) / 2.0; // Vermeide Division durch Null
	return x1 + (x2 - x1) * (a_target - a_val1) / (a_val2 - a_val1);
}


// Interpolation von ln(K) zwischen Bubble- und Dew-Punkten
Eigen::VectorXd interpolateK(
	double P, double Pdew, double Pbub,
	const Eigen::VectorXd& Kdew,
	const Eigen::VectorXd& Kbub,
	double azeoTol = 1e-4)
{
	if (Kdew.size() != Kbub.size()) {
		throw std::runtime_error("Kdew and Kbub have different sizes!");
	}

	const int N = Kdew.size();
	Eigen::VectorXd K(N);

	// Interpolation factor in log(P) space
	double theta = (std::log(Pdew) - std::log(P)) /
		(std::log(Pdew) - std::log(Pbub));

	bool nearAzeotrope = true;

	for (int i = 0; i < N; ++i) {
		double lnKdew = std::log(std::max(Kdew[i], 1e-12));
		double lnKbub = std::log(std::max(Kbub[i], 1e-12));

		double lnK = theta * lnKbub + (1.0 - theta) * lnKdew;
		K[i] = std::exp(lnK);

		if (std::fabs(K[i] - 1.0) > azeoTol) {
			nearAzeotrope = false;
		}
	}

	if (nearAzeotrope) {
		K.setOnes();
	}

	return K;
}


// Funktion zur Berechnung aller Schnittpunkte und zugehöriger x-Werte mit Konsistenzprüfung
std::vector<std::pair<std::pair<double, double>, std::vector<double>>> findAllIntersectionsWithX(
	const std::vector<double>& a1, const std::vector<double>& a2, const std::vector<double>& x) {
	if (a1.size() != a2.size() || a1.size() != x.size() || a1.size() < 2) {
		throw std::invalid_argument("Vektoren müssen gleich lang sein und mindestens 2 Elemente enthalten");
	}

	std::vector<std::pair<std::pair<double, double>, std::vector<double>>> intersectionsWithX;
	size_t n = a1.size();
	const double tolerance = 1e-3; // Toleranz für die Übereinstimmung der x-Werte

	// Prüfen aller Segmentpaare
	for (size_t i = 0; i < n - 1; ++i) {
		for (size_t j = 0; j < n - 1; ++j) {
			if (j == i || (j == i + 1 && j < n - 1)) continue; // Überspringe identische oder benachbarte Segmente
			std::pair<double, double> intersection;
			if (findSegmentIntersection(
				a1[i], a2[i], a1[i + 1], a2[i + 1],
				a1[j], a2[j], a1[j + 1], a2[j + 1],
				intersection)) {
				// Prüfe auf Duplikate mit Toleranz
				bool duplicate = false;
				for (const auto& existing : intersectionsWithX) {
					if (std::abs(existing.first.first - intersection.first) < 1e-5 &&
						std::abs(existing.first.second - intersection.second) < 1e-5) {
						duplicate = true;
						break;
					}
				}
				if (!duplicate) {
					double a1_schnitt = intersection.first;
					double a2_schnitt = intersection.second;
					std::vector<double> x_a2_values, x_a1_values;

					// Finde alle Intervalle für a2_schnitt
					for (size_t k = 0; k < n - 1; ++k) {
						if ((a2[k] <= a2_schnitt && a2[k + 1] >= a2_schnitt) ||
							(a2[k] >= a2_schnitt && a2[k + 1] <= a2_schnitt)) {
							double x_schnitt = interpolateX(x[k], x[k + 1], a2[k], a2[k + 1], a2_schnitt);
							x_a2_values.push_back(x_schnitt);
						}
					}

					// Finde alle Intervalle für a1_schnitt
					for (size_t k = 0; k < n - 1; ++k) {
						if ((a1[k] <= a1_schnitt && a1[k + 1] >= a1_schnitt) ||
							(a1[k] >= a1_schnitt && a1[k + 1] <= a1_schnitt)) {
							double x_schnitt = interpolateX(x[k], x[k + 1], a1[k], a1[k + 1], a1_schnitt);
							x_a1_values.push_back(x_schnitt);
						}
					}

					// Prüfe Konsistenz zwischen x_a2_values und x_a1_values
					bool consistent = false;
					if (!x_a2_values.empty() && !x_a1_values.empty() &&
						x_a2_values.size() >= 2 && x_a1_values.size() >= 2) {
						double diff_front = std::abs(x_a1_values.front() - x_a2_values.front());
						double diff_back = std::abs(x_a1_values.back() - x_a2_values.back());
						if (diff_front < tolerance && diff_back < tolerance) {
							consistent = true;
						}
					}

					if (consistent && !x_a2_values.empty()) {
						intersectionsWithX.emplace_back(intersection, x_a2_values);
					}
				}
			}
		}
	}

	return intersectionsWithX;
}
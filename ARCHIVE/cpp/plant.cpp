#include "plant.h"
// #include <boost/math/special_functions/erf.hpp>
using namespace std;
// using namespace boost::math;

double plant(double x){

        extern double Fs;
	double dt = 1/Fs;

	//const double a = 1.732814112;
	//const double b = 42.6005776;
	//const double G = 31.33967453;
	//const double G = 1335.088236;
	//const double c = 0.558289198;

	/*const double a = 3.044128029;
	const double b = 47.17286919;
	const double G = 26.94948742 * b;
	const double c = 0.011879839;
	const double ee = 90;*/

	const double a = 1.822724504;
	const double b = 40.79680257;
	const double G = 74.03109726 * b;
	//	const double c = 0.00336088;
	//	const double ee = 110;

	static double y = 0.0;
	static double r = 0.0;

	double dy = r;
	double dr = G*x - a*r - b*y;

	y += dt*dy;
	r += dt*dr;

	//	double val = ee*erf(c*y);
	//	return val;
	return y;
}

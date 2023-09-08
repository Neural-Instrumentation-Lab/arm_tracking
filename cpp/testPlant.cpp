#include <iostream>
#include <fstream>
#include <cmath>
// #include "windows.h"
// #include <boost/math/special_functions/erf.hpp>
#include "plant.h"
using namespace std;
// using namespace boost::math;
double Fs = 256;

int main(int argc, char **argv){
	int i = 0;
	double t;
	double dt;
	double w = 1.0;
	double tmax = 20;
	int nSamples;

	if (argc > 1)
	  w = atof(argv[1]);
	if (argc > 2)
	  Fs = atof(argv[2]);
	cout << "Fs = " << Fs << endl;

	dt = 1/Fs;
	nSamples = (int)(Fs*tmax);
	double *x = new double [nSamples];
	double *y = new double [nSamples];

	//ofstream fid1("in3.txt");
	//ofstream fid2("out3.txt");

	//	ifstream vfile("sinout_8-10_r1.txt");
	
	ofstream fid1("rin.bin", ios::out | ios::binary);
	ofstream fid2("rout.bin",ios::out | ios::binary);

	for (i=0;i<nSamples;i++){
	  t = i*dt;
	  x[i] = cos(w*t);
	}
	// create input
	// for (int i=0;i<nSamples;i++){
	// 	vfile >> t >> x[i];
	// 	x[i] -= 4;}

	// simulate output
	y[0] = 0;
	for (i=0;i<nSamples-1;i++)
		y[i+1] = plant(x[i]);

	fid1.write((char *)x,nSamples*sizeof(double));
	fid2.write((char *)&Fs,sizeof(double));
	fid2.write((char *)y,nSamples*sizeof(double));

	fid1.close();
	fid2.close();
	delete [] x;
	delete [] y;
	return 0;
}

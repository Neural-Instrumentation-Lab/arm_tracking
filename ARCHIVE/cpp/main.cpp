/*
 * main.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: iobeid
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
//#include "engine.h"
#include "cerebellum.h"
#include <boost/math/special_functions/erf.hpp>
using namespace std;
using namespace boost::math;

#define FS 40000
double dt = 1.0/FS;



int main(int argc,char *argv[]){
	const int nCerebellums = 10;
	cerebellum c[nCerebellums];
	double **centers;
	int nDims = 1;
	int nCenters = 8;
	double tmax = 20.0;
	int nPts =(int)(tmax*FS);
	double freq = 1.0;
	int i,j;
	bool matFlag = false;

	double *weightArchive = new double[nPts * nCenters];

	cout << "starting\n";

	/*if (argc > 1)
		if (strcmp(argv[1],"m")==0){
			cout << "setting matFlag" << endl;
			matFlag = true;
		}

	if (!matFlag)
		cout << "matFlag not set" << endl;*/

	// allocate memory for centers
	centers = new double* [nCenters];
	for (i=0;i<nCenters;i++)
		centers[i] = new double [nDims];

	// define the centers
	double ctrMx = 1;
	double ctrMn = -ctrMx;
	double ctrSpacing = (ctrMx-ctrMn)/(nCenters-1);
	for (i=0;i<nCenters;i++)
		centers[i][0] = ctrMn + i*ctrSpacing;
	double sg = ctrSpacing; // 0.125;

	// initialize the various cerebellums
	for (i=0;i<nCerebellums;i++)
		c[i].init(nCenters,centers,nDims,sg);

	// define a trajectory
	double *vDesired  = new double [nPts];
	double *t         = new double [nPts];
	double *vRealized = new double [nPts];

	vDesired[i] = cos(2*M_PI*freq*t[i]);

	// execute the trajectory
	double vCommand, err;
	double stim;
	for (i=nCerebellums;i<nPts;i++){

		// collect weights into Archive
		for (j=0;j<nCenters;j++)
			weightArchive[i*nCenters + j] = c[0].getWeight(j);

		stim = 0;
		for (j=0;j<nCerebellums;j++)
			stim += c[j].stimulate(&(vDesired[i-j]));
		vCommand = vDesired[i] + stim;
		vRealized[i] = plant(vCommand);
		err = vRealized[i] - vDesired[i];

		for(j=0;j<nCerebellums;j++)
			c[j].updateErrors(err);

	}


	//if (matFlag){
	//	Engine *ep = engOpen(NULL);

	//	mxArray *pmx_t = mxCreateDoubleMatrix(nPts,1,mxREAL);
	//	mxArray *pmx_x = mxCreateDoubleMatrix(nPts,1,mxREAL);
	//	mxArray *pmx_y = mxCreateDoubleMatrix(nPts,1,mxREAL);
	//	mxArray *pmx_w = mxCreateDoubleMatrix(nPts*nCenters, 1, mxREAL);
	//	mxArray *pmx_m = mxCreateDoubleMatrix(1   ,1,mxREAL);

	//	double *pmt = mxGetPr(pmx_t);
	//	double *pmx = mxGetPr(pmx_x);
	//	double *pmy = mxGetPr(pmx_y);
	//	double *pmw = mxGetPr(pmx_w);
	//	double *pmm = mxGetPr(pmx_m);

	//	pmm[0] = tmax;
	//	for (i=0;i<nPts;i++){
	//		pmt[i] = t[i];
	//		pmx[i] = vDesired[i];
	//		pmy[i] = vRealized[i];

	//		for (j=0;j<nCenters;j++)
	//			pmw[i*nCenters + j]=weightArchive[i*nCenters + j];

	//	}

	//	engPutVariable(ep,"t",pmx_t);
	//	engPutVariable(ep,"x",pmx_x);
	//	engPutVariable(ep,"y",pmx_y);
	//	engPutVariable(ep,"tmax",pmx_m);
	//	engPutVariable(ep,"w",pmx_w);


	//	engEvalString(ep,"w=reshape(w,8,[]);");
	//	engEvalString(ep,"save weightData w");

	//	engEvalString(ep,"plot(t,x,t,y); axis([0 tmax -2 2])");
	//	engEvalString(ep,"legend('x','y');");
	//	engEvalString(ep,"waitfor(gcf);");

	//	engClose(ep);
	//}

	//for(i=0;i<nCenters;i++)
	//	delete [] centers[i];
	//delete [] centers;
	//delete [] weightArchive;

	//cout << "exiting\n";

	return 0;
}


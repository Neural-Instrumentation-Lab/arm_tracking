/*
 * rbf.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: iobeid
 */

#include "rbf.h"
#include <cmath>
#include "windows.h"

rbf::rbf(){
  nDimensions = 0;
  center = NULL;
  sigma = 0;
}

rbf::rbf(int nD, double *ctr, double sg){
  init(nD,ctr,sg);
}

void rbf::init(int nD, double *ctr, double sg){
  int i;
  nDimensions = nD;
  center = new double[nD];
  for (i=0;i<nD;i++)
    center[i] = ctr[i];
  sigma = sg;
}

rbf::~rbf(){
  delete [] center;
}

double rbf::activation(double *pt){
  int i;
  double s = 0;
  for(i=0;i<nDimensions;i++)
    s += pow(pt[i]-center[i] , 2);
  s = -s/(2*sigma*sigma);
  s = exp(s);
  return s;
}

double rbf::getCenter(int i){
  return center[i];
}

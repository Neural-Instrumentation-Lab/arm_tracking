#include <iostream>
#include <iomanip>
#include "engine.h"
#include <cmath>

using namespace std;

#define FS 40000
double dt = 1.0/FS;
Engine *ep = engOpen(NULL);

double plant(double x){
  double fc = 2;
  static double y = 0;
  double dy = (x-y)*2*M_PI*fc;
  y += dt*dy;
  return y;
}


int main(){
  int nPts = 10*FS;
  double *x = new double [nPts];
  double *y = new double [nPts];
  double *t = new double [nPts];
  double f = 1;
  int i;

  for (i=0;i<nPts;i++){
    t[i] = i * dt;
    x[i] = cos(2*M_PI*f*t[i]);
    y[i] = plant(x[i]);
    if ((i%100)==0)
      cout << i << "\t" << t[i] << "\t" << x[i] << "\t"<< y[i] << endl;
  }  

  mxArray *pmx_t = mxCreateDoubleMatrix(nPts,1,mxREAL);
  mxArray *pmx_x = mxCreateDoubleMatrix(nPts,1,mxREAL);
  mxArray *pmx_y = mxCreateDoubleMatrix(nPts,1,mxREAL);
  double *pmt = mxGetPr(pmx_t);
  double *pmx = mxGetPr(pmx_x);
  double *pmy = mxGetPr(pmx_y);

  for (i=0;i<nPts;i++){
    pmt[i] = t[i];
    pmx[i] = x[i];
    pmy[i] = y[i];
  }
  engPutVariable(ep,"t",pmx_t);
  engPutVariable(ep,"x",pmx_x);
  engPutVariable(ep,"y",pmx_y);
  engEvalString(ep,"plot(t,x,t,y);");
  engEvalString(ep,"title(sprintf('%.5f',max(y)));");
  engEvalString(ep,"waitforbuttonpress;");

  engClose(ep);

 return 0;
}

#include "kmeans.h"

int main(int argc, char **argv)
{

  KMeans *km = new KMeans();
  
  km->setPointsNum(1000);
  km->setRoles(3);
  km->setPoints();
  km->saveAs("origin.jpg");
  
  km->solve();
  
  km->saveAs("result.jpg");
  
  delete km;
  
  return 0;

}

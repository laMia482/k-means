#ifndef K_MEANS_H_
#define K_MEANS_H_

#include <map>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

typedef int KInt;
typedef float KFloat;
typedef int KCate;
typedef struct
{
  KInt x;
  KInt y;
  KCate c;
}KPoint;

typedef std::vector<KPoint> KPoints;

class KMeans
{
public:
  KMeans(void);
  KMeans(const KInt &, const KInt &);
  ~KMeans(void);
  void setPointsNum(const KInt &);
  void setRoles(const KInt &);
  void setPoints(void);
  void solve(void);
  void saveAs(const std::string &);
  void printPoints(void);
  
private:
  KPoint generatePoint(void);
  void addPoint(const KPoint &);
  template<class T> 
  T randGenerate(const T &, const T &);
  void drawPoint(cv::Mat &, const KPoint &);
  void paintOn(cv::Mat &);
  KFloat calcJourney(const KPoint &, const KPoint &);
  KInt selectMin(const KInt &);
  void dividePoints(void);
  KPoint calcAvgPoints(const KInt &);
  void updateCenter(void);
  bool solved(void);
  
  KInt m_PointsNum;
  KInt m_PointsCate;
  KPoints m_PointsCenter, m_LastPointsCenter;
  KPoints m_Points;
  KInt m_Round;
  std::map<KInt, std::vector<KFloat> > m_Jour;
  KInt m_CurIter, m_FinalIter;
  KFloat m_CurLoss, m_LastLoss;
  std::map<KInt, cv::Scalar> m_Color;
};


#endif // ! K_MEANS_H_
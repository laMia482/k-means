#include "kmeans.h"

#ifdef K_MEANS_H_

KMeans::KMeans(void)
{
  google::InitGoogleLogging(gen);
  srand(time(NULL));
  m_Round = 500;
  m_CurIter = 0;  m_FinalIter = 1000;
  m_CurLoss = 0.0f; m_LastLoss = 0.0f;
}

KMeans::KMeans(const KInt &pointsNum, const KInt &pointsCate)
{
  KMeans();
  setPointsNum(pointsNum);
  setRoles(pointsCate);
}

KMeans::~KMeans(void)
{
  google::ShutdownGoogleLogging();
}

void KMeans::setPointsNum(const KInt &pointsNum)
{
  m_PointsNum = pointsNum;
}

void KMeans::setRoles(const KInt &pointsCate)
{
  m_PointsCate = pointsCate;
  m_Color[0] = cv::Scalar(0, 0, 255);
  m_Color[1] = cv::Scalar(0, 255, 0);
  m_Color[2] = cv::Scalar(255, 0, 0);
  m_Color[3] = cv::Scalar(255, 255, 255);
  m_PointsCenter.clear();
  m_LastPointsCenter.clear();
  for(KInt i=0;i<m_PointsCate;++i)
  {
    auto kp = generatePoint();
    kp.c = 3;
    m_PointsCenter.push_back(kp);
    m_LastPointsCenter.push_back(kp);
  }
}

void KMeans::addPoint(const KPoint &kp)
{
  m_Points.push_back(kp);
}

template<class T> 
T KMeans::randGenerate(const T &A, const T &B)
{
  auto res = A + B * 1. *rand() /  RAND_MAX;
  return res;
}

KPoint KMeans::generatePoint(void)
{
  auto x = randGenerate(10, m_Round - 10);
  auto y = randGenerate(10, m_Round - 10);
  auto c = randGenerate(0, m_PointsCate);
  KPoint pt;
  pt.x = x; pt.y = y; pt.c = c;
  return pt;
}

void KMeans::setPoints(void)
{
  m_Points.clear();
  for(KInt i=0;i<m_PointsNum;++i)
  {
    auto kp = generatePoint();
    addPoint(kp);
  }
}

KFloat KMeans::calcJourney(const KPoint &x, const KPoint &y)
{
  auto res = std::sqrt(std::pow(x.x - y.x, 2) + std::pow(x.y - y.y, 2));
  return res;
}

KInt KMeans::selectMin(const KInt &i)
{
  KInt res = -1;
  KFloat jour = m_Round^2;
  m_Jour[i].resize(m_PointsCate);
  for(KInt c=0;c<m_PointsCate;++c)
  {
    m_Jour[i][c] = calcJourney(m_Points[i], m_PointsCenter[c]);
    // LOG(INFO) << "[" << m_Points[i].x << ", " << m_Points[i].y << "], " \
              << "[" << m_PointsCenter[c].x << ", " << m_PointsCenter[c].y << "], " \
              << "jour: " << m_Jour[i][c];
    if(m_Jour[i][c] < jour)
    {
      jour = m_Jour[i][c];
      res = c;
    }
  }
  // LOG(INFO) << "res: " << res;
  return res;
}

// divide points by the distance to each center
void KMeans::dividePoints(void)
{
  for(KInt i=0;i<m_PointsNum;++i)
    m_Points[i].c = selectMin(i);
}

KPoint KMeans::calcAvgPoints(const KInt &c)
{
  long x = 0, y = 0, n = 0;
  for(KInt i=0;i<m_PointsNum;++i)
  {
    auto kp = m_Points[i];
    if(c == kp.c)
    {
      x += kp.x;
      y += kp.y;
      ++n;
    }
  }
  if(!n)
    return m_PointsCenter[c];
  KPoint kp;
  kp.x = 1. * x / n;
  kp.y = 1. * y / n;
  kp.c = 3;
  // LOG(INFO) << "center[" << c << "] = [" << kp.x << ", " << kp.y << "]";
  return kp;
}

// update category centers by loss
void KMeans::updateCenter(void)
{
  for(int c=0;c<m_PointsCate;++c)
    m_PointsCenter[c] = calcAvgPoints(c);
  ++m_CurIter;
}

// see if update go on
bool KMeans::solved(void)
{
  if(m_CurIter > m_FinalIter)
    return true;
  m_CurLoss = 0.0f;
  for(KInt c=0;c<m_PointsCate;++c)
    m_CurLoss += calcJourney(m_PointsCenter[c], m_LastPointsCenter[c]);
  if(m_CurLoss == m_LastLoss)
    return true;
  m_LastLoss = m_CurLoss;
  return false;
}

void KMeans::solve(void)
{
  while(true)
  {
    dividePoints();
    updateCenter();
    if(solved())
      break;
  }
  LOG(INFO) << "DONE FOR K-MEANS, ITER: " << m_CurIter << " LOSS: " << m_CurLoss;
}

void KMeans::drawPoint(cv::Mat &image, const KPoint &kp)
{
  auto pt = cv::Point(kp.x, kp.y);
  cv::circle(image, pt, 3, m_Color[kp.c], 3);
}

void KMeans::paintOn(cv::Mat &image)
{
  for(KInt num=0;num<m_PointsNum;++num)
    drawPoint(image, m_Points[num]);
  for(KInt num=0;num<m_PointsCate;++num)
    cv::circle(image, cv::Point(m_PointsCenter[num].x, m_PointsCenter[num].y), 3, m_Color[m_PointsCenter[num].c], 3);
}

void KMeans::saveAs(const std::string &fileName)
{
  cv::Mat image = cv::Mat::zeros(m_Round, m_Round, CV_8UC3);
  paintOn(image);
  cv::imwrite(fileName, image);
}

void KMeans::printPoints(void)
{
  for(KInt i=0;i<m_PointsNum;++i)
    LOG(INFO) << "Point[" << i << "] = [" << m_Points[i].x << ", " << m_Points[i].y << "]: " << m_Points[i].c;
}












#endif // ! K_MEANS_H_

#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <boost/math/distributions/normal.hpp>


#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;
double distance2D(cv::Rect&,cv::Rect&);

// Function to get medium
double getMedian(std::vector<double>& vec){
    std::sort(vec.begin(),vec.end());
    int vecSize=vec.size();
    if(vecSize%2)
        return vec[(vecSize+1)/2];
    else
        return (*(vec.begin()+(vecSize/2 - 1))+*(vec.begin() + vecSize/2));
}

// Function to give vlaues of 0.025 quantile and 0.975 quantile  
std::pair<double,double> getIQR(std::vector<double> vec){
    double q1=0;
    double q3=0;
    std::vector<double> firstHalf;
    std::vector<double> secondHalf;

    std::sort(vec.begin(),vec.end());
    int vecSize=vec.size();
    if(vecSize%2){ //odd number of samples
        int medianPoint=(vecSize+1)/2;
        firstHalf=std::vector<double>(vec.begin(),vec.begin()+(medianPoint-1));
        secondHalf=std::vector<double>(vec.begin()+ medianPoint+1,vec.end());
    }
    else{   // even number of sample
        firstHalf=std::vector<double>(vec.begin(),vec.begin()+(vecSize/2 - 1));
        secondHalf=std::vector<double>(vec.begin()+vecSize/2,vec.end());

    }

    q1=getMedian(firstHalf);
    q3=getMedian(secondHalf);

    std::pair<double,double>(q1,q3);

}

template<typename T, typename U>
void printMap(std::map<T,U> testMap);

void printVector(std::vector<int>& vec);

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    /*

    This function calculate the 2D Eculidian distance between the matches. Then, it find IQR for the distances distribution
    Only, if the distance is located within IQR, it will be taken into account.


    */
    
    
    std::vector<double> distances; // to save the eculedian distances between all the matching points 
    
    for(auto& match:kptMatches)
        distances.push_back(cv::norm(kptsCurr[match.queryIdx].pt-kptsPrev[match.trainIdx].pt));

    std::pair<double,double> iqr=getIQR(distances);
    double q1=iqr.first;
    double q3=iqr.second;
    
    double eculedianDist=0; //Eculedian distance between two matching point
    
    for(auto& match:kptMatches){

        cv::KeyPoint srcKeyPnt=kptsCurr[match.queryIdx];
        
        cv::KeyPoint refKeyPnt=kptsPrev[match.trainIdx];
        if(boundingBox.roi.contains(srcKeyPnt.pt)){
            eculedianDist=cv::norm(refKeyPnt.pt-srcKeyPnt.pt); //calcualte the Eculedian distance
            
            if((eculedianDist>=q1) && (eculedianDist<=q3)){ // check if the distance is located within IQR
                
                boundingBox.kptMatches.emplace_back(match);
            }
        }
        
    }
    
    

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> ratioDistances;
    cv::KeyPoint kptCurrPnt1;
    cv::KeyPoint kptPrevPnt1;
    cv::KeyPoint kptCurrPnt2;
    cv::KeyPoint kptPrevPnt2;
    double distanceCurr;
    double distancePrev;
    double minDistance= 100;
    double dt=1/frameRate;
    
    for(auto itr1=kptMatches.begin();itr1!=kptMatches.end()-1;++itr1){
        kptCurrPnt1=kptsCurr[itr1->trainIdx];
        kptPrevPnt1=kptsPrev[itr1->queryIdx];

        for(auto itr2=kptMatches.begin();itr2!=kptMatches.end();++itr2){
            kptCurrPnt2=kptsCurr[itr2->trainIdx];
            kptPrevPnt2=kptsPrev[itr2->queryIdx];

            distanceCurr=cv::norm(kptCurrPnt1.pt - kptCurrPnt2.pt);
            
            distancePrev=cv::norm(kptPrevPnt1.pt - kptPrevPnt2.pt);
            

            if( distancePrev>=std::numeric_limits<double>::epsilon() && distanceCurr>=minDistance)
                ratioDistances.push_back(distanceCurr/distancePrev);

        }
    }

    if(ratioDistances.empty()){
        TTC=NAN;
        std::cout<<"empty"<<std::endl;
        return;
    }
    
    double medianDistance=getMedian(ratioDistances);
    TTC=-dt/(1-medianDistance);

    std::cout<<"TTC using camera is "<<TTC<<"secs"<<std::endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    

    std::size_t nPrev=lidarPointsPrev.size()/2;
    std::size_t nCurr=lidarPointsCurr.size()/2;
 
    std::vector<double> xPrev;
    std::vector<double> xCurr;

    for(auto itrPrev=lidarPointsPrev.begin();itrPrev!=lidarPointsPrev.end();++itrPrev)
        xPrev.push_back(itrPrev->x);
    
    for(auto itrCurr=lidarPointsCurr.begin();itrCurr!=lidarPointsCurr.end();++itrCurr)
        xCurr.push_back(itrCurr->x);
    
    TTC=getMedian(xCurr)/((getMedian(xPrev)-getMedian(xCurr))*frameRate);

    std::cout<<"TTC using lidar is "<<TTC<<"secs"<<std::endl;
        
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    /*

    The idea here is to benefit from the high camera data rate and compare the distance between 
    each bounding box in the prev frame and the current frame. The two with the least distance are the matched ones.




    */   
    
    cv::KeyPoint prevFramePt;
    cv::KeyPoint currentFramePt;
    
    for(auto match:matches){
        prevFramePt=prevFrame.keypoints[match.trainIdx];
        currentFramePt=currFrame.keypoints[match.queryIdx];

        for(auto itr=prevFrame.boundingBoxes.begin();itr!=prevFrame.boundingBoxes.end();++itr){

            itr->crossMatchPoints=std::vector<double>(currFrame.boundingBoxes.size(),0);
            if(itr->roi.contains(prevFramePt.pt)){
                itr->keypoints.emplace_back(prevFramePt);
                itr->kptMatches.emplace_back(match);
            }

                for(auto itr_2=currFrame.boundingBoxes.begin();itr_2!=currFrame.boundingBoxes.end();++itr_2){

                    if(itr_2->roi.contains(currentFramePt.pt)){
                        itr_2->keypoints.emplace_back(currentFramePt);
                        itr_2->kptMatches.emplace_back(match);
                    }
                    double distanceValue=distance2D(itr->roi,itr_2->roi);
                    itr->crossMatchPoints[itr_2->boxID]=distanceValue; 
                    



                } 
            

        }

    }

    for(auto itr=prevFrame.boundingBoxes.begin();itr!=prevFrame.boundingBoxes.end();++itr){
        auto matchedBox=std::distance(itr->crossMatchPoints.begin(),std::min_element(itr->crossMatchPoints.begin(),itr->crossMatchPoints.end()));
                //std::cout<<"min Matched"<<matchedBox<<std::endl;
        bbBestMatches[itr->boxID]=matchedBox;
    }
           
        
}


double distance2D(cv::Rect& box1,cv::Rect& box2){


    cv::Point2d box1CenterPt(box1.x+box1.width/2,box1.y+box1.height/2);

    cv::Point2d box2CenterPt(box2.x+box2.width/2,box2.y+box2.height/2);

    return cv::norm(box1CenterPt-box2CenterPt);


}

template<typename T, typename U>
void printMap(std::map<T,U> testMap){
    for(auto itr:testMap){
        std::cout<< itr.first <<"--->"<< itr.second <<std::endl;
    }
}

 void printVector(std::vector<int>& vec){
       for(auto itr=vec.begin();itr!=vec.end();++itr)
            std::cout<<*itr<<"\t";
        std::cout<<"\n";

   }
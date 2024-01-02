#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;

Mat curr_frame;                             //used for current frame of video
Mat imgGray, imgBlur, imgCanny, dst, mask;  //used for image processing in main
Mat imgHandGray, imgPersonGray;
RNG rng(12345);
int thresh = 100;
int hmin = 0, smin = 110, vmin = 153;
int hmax = 19, smax = 240, vmax = 255;
int lowthreshold = 0;
const int max_lowThreshold = 100;
const int ratio =3;
const int kernel_size = 3;

void getHandContours(int, void* );
void getPersonContours(int, void* );

int main()
{

    //define id# of camera
    VideoCapture cam(0);

    //grab size of camera
    //Size sz = img.size();
    //cout << "heigh & width of captured frame " << sz.height << " x " << sz.width;

    //trackbars for slider
    /*
    namedWindow("Trackbars", (640, 200));
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("sat min", "Trackbars", &smin, 255);
    createTrackbar("sat max", "Trackbars", &smax, 255);
    createTrackbar("V min", "Trackbars", &vmin, 255);
    createTrackbar("V max", "Trackbars", &vmax, 255);
    */
    //frame counter
    int frame = 1;
    const int max_thresh = 255;
    //namedWindow( "Source_window" );
    //createTrackbar("Canny Thresh", "Source_window", &thresh, max_thresh, getHandContours);

    while (true) {

        bool bSuccess = cam.read(curr_frame);

        if (!bSuccess){
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        //update counter
        ++frame;

        //reduce noise by applying grayscale and Gausian Blur
        cvtColor(curr_frame, imgGray, COLOR_BGR2GRAY);
        GaussianBlur(imgGray, imgBlur, Size(3, 3), 0, 0, BORDER_DEFAULT);

        //apply an Otsu Inv threshold to the image
        threshold(imgGray, imgBlur, 0 ,255, THRESH_OTSU + THRESH_BINARY_INV);
    
        imgHandGray = imgGray.clone();
        imgPersonGray = imgGray.clone();

        //thread for hand processing
        thread th1(getHandContours, 0, 0);

        //thread for person processing
        thread th1(getPersonContours, 0, 0);

        imshow("Source_window", curr_frame);
        waitKey(1);
    }

    destroyAllWindows();
    return 0;
}

//function to process and scarpe the image for hand signs
void getHandContours(int, void*)
{
    Mat canny_output;
    Canny(imgHandGray, imgCanny, 100, 100*2);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(imgCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>>hull ( contours.size() );
    for(size_t i = 0; i < contours.size(); i++){
        convexHull( contours[i], hull[i]);
    }

    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3);
    for( size_t i = 0; i < contours.size(); i++){
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours(drawing, contours, i, color);
        drawContours(drawing, hull, i, color);

    }
    imshow("Contours", drawing);
}

//function to process the image to find the contour of a person
void getPersonContours(int, void*)
{
    Mat canny_output;
    Canny(imgPersonGray, imgCanny, 100, 100*2);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(imgCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>>hull ( contours.size() );
    for(size_t i = 0; i < contours.size(); i++){
        convexHull( contours[i], hull[i]);
    }

    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3);
    for( size_t i = 0; i < contours.size(); i++){
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours(drawing, contours, i, color);
        drawContours(drawing, hull, i, color);

    }
    imshow("Contours", drawing);
}

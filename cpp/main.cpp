#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv )
{

    cv::Mat image;
    image = cv::imread("C:/Users/elkin/Documents/University Files/Tenth Semester/Computer Vision/computer-vision-dojo/cpp/assets/imgs/Foto_mia_baxter.jpeg");
    if ( !image.data )
    {
        std::cout << "No image data" << std::endl;
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    return 0;
}
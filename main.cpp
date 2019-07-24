#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade, eyes_cascade;

void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_RGB2GRAY);
	equalizeHist(frame_gray, frame_gray);
	vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 0), 4);

		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(frame_gray(faces[i]), eyes);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(0, 0, 255), 4);
		}
	}
	imshow("Capture - Face detection", frame);
}

int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{face_cascade|lbpcascade_frontalface_improved.xml|Path to face cascade.}"
		"{eyes_cascade|haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.printMessage();
	
	if (!face_cascade.load(parser.get<String>("face_cascade")))
		cout << "Error loading face cascade\n";
	if (!eyes_cascade.load(parser.get<String>("eyes_cascade")))
		cout << "Error loading eyes cascade\n";

	int camera_device = parser.get<int>("camera");

	VideoCapture capture;
	capture.open(camera_device);

	if (!capture.isOpened())
		cout << "Error opening video capture\n";

	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "No captured frame -- Break!\n";
			break;
		}
		detectAndDisplay(frame);
		if (waitKey(10) == 27)
			break;
	}
	return 0;
}

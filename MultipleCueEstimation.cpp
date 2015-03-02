#include "MultipleCueEstimation.h"

using namespace cv;

MultipleCueEstimation::MultipleCueEstimation(Mat *img1, Mat *img2, int comp) {
    image_1_color = img1->clone();
    image_2_color = img2->clone();
    compareWithGroundTruth = false;
    computations = comp;
}

MultipleCueEstimation::MultipleCueEstimation(Mat *img1, Mat *img2, int comp, Mat *F_groudtruth)
{
    image_1_color = img1->clone();
    image_2_color = img2->clone();
    compareWithGroundTruth = false;
    computations = comp;
    compareWithGroundTruth = true;
    Fgt = F_groudtruth->clone();
}

void MultipleCueEstimation::run() {
    std::vector<FEstimationMethod> estimations;
    if (checkData()) {
        if(computations & F_FROM_POINTS) {
            estimations.push_back(*calcFfromPoints());
        }
        if(computations & F_FROM_LINES) {
            estimations.push_back(*calcFfromLines());
        }
        if(computations & F_FROM_PLANES) {
            estimations.push_back(*calcFfromPlanes());
        }

        if (compareWithGroundTruth) {
            std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1,x2, Fgt, image_1, image_2, "F_groundtruth");
            }
        }

        //double error = epipolarSADError(Fgt, x1, x2);

        for (std::vector<FEstimationMethod>::iterator it = estimations.begin() ; it != estimations.end(); ++it) {
            std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
            if (compareWithGroundTruth) {
                double error1 = epipolarLineDistanceError(Fgt, it->getF(), image_1, NUM_SAMPLES_F_COMARATION);
                double error2 = squaredError(Fgt, it->getF());
                std::cout << "Random sample epipolar error: " << error1 << ", Squated distance: " << error2 << std::endl;
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
            }
        }

        waitKey(0);
    }

}

int MultipleCueEstimation::checkData() {
    if(!image_1_color.data || !image_2_color.data)
    {
        std::cerr << "No image data!" << std::endl;
        return 0;
    }

    if(compareWithGroundTruth && !Fgt.data) {
        std::cerr << "No ground truth data!" << std::endl;
        return 0;
    }

    if(image_1_color.channels() < 1) {
        cvtColor(image_1_color, image_1, CV_BGR2GRAY);
        cvtColor(image_2_color, image_2, CV_BGR2GRAY);
    } else {
        image_1 = image_1_color;
        image_2 = image_2_color;
    }

    if(VISUAL_DEBUG) {
        showImage("Image 1 original", image_1_color);
        showImage("Image 2 original", image_2_color);
    }

    return 1;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPoints() {
    FEstimatorPoints* estimatorPoints = new FEstimatorPoints(image_1, image_2, image_1_color, image_2_color, "F_points");
    estimatorPoints->compute();
    x1 = estimatorPoints->getX1();
    x2 = estimatorPoints->getX2();
    return estimatorPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    FEstimatorLines* estomatorLines = new FEstimatorLines(image_1, image_2, image_1_color, image_2_color, "F_lines");
    estomatorLines->compute();
    return estomatorLines;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPlanes() {    // From: 1. two Homographies (one in each image), 2. Planes as additinal point information (point-plane dualism)
    FEstimatorPlanes* estomatorPlanes = new FEstimatorPlanes(image_1, image_2, image_1_color, image_2_color, "F_Planes");
    estomatorPlanes->compute();
    return estomatorPlanes;
}

FEstimationMethod* MultipleCueEstimation::calcFfromConics() {    // Maybe: something with vanishing points v1*w*v2=0 (Hartley, Zissarmen p. 235ff)

}

FEstimationMethod* MultipleCueEstimation::calcFfromCurves() {    // First derivative of corresponding curves are gradients to the epipolar lines

}

Mat MultipleCueEstimation::refineF() {    //Reduce error of F AFTER computing it seperatly form different sources

}




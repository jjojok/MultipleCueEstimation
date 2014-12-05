#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MCE.h"

int main(int argc, char** argv )
{
    freopen( "error.txt", "w", stderr );
    MCE* mce = new MCE();
    mce->arguments = argc;
    if (mce->arguments == 6) {
        mce->path_P1 = argv[4];
        mce->path_P2 = argv[5];
        mce->compareWithGroundTruth = true;
    }
    mce->path_img1 = argv[1];
    mce->path_img2 = argv[2];
    mce->computations = std::atoi(argv[3]);
    mce->run();
}

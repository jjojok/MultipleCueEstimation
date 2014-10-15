#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MCE.h"

int main(int argc, char** argv )
{
    freopen( "error.txt", "w", stderr );
    MCE* mce = new MCE(argc, argv);
    mce->run();
}

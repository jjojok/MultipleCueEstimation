#!/bin/sh
if [ ! -d "Results" ]
then
	mkdir "Results"
fi

PREFIX="/home/johannes/MultipleCueEstimation/Datasets/"

OUT="Results/resultList.csv"

#table header
echo "first image, second image, featureCountCombined, trueFeatureCountCombined, Fgt: trueSampsonErr, Fgt: trueErrorStdDev, Fgt: trueRootSampsonErr, Fgt: trueRootErrorStdDev, refined F: inlierCountCombined, refined F: trueInlierCountCombined, refined F: sampsonErrCombined, refined F: trueSampsonErr, refined F: trueSampsonErrStdDev, refined F: trueRootSampsonErr, refined F: trueRootSampsonErrStdDev, F_Points: featureCountGood, F_Points: trueFeatureCountGood, F_Points: featureCountComplete, F_Points: trueFeatureCountComplete, F_Points: inlierCountOwnGood, F_Points: trueInlierCountOwnGood, F_Points: inlierCountOwnComplete, F_Points: trueInlierCountOwnComplete, F_Points: inlierCountCombined, F_Points: trueInlierCountCombined, F_Points: sampsonErrOwn, F_Points: sampsonErrComplete, F_Points: sampsonErrCombined, F_Points: sampsonErrStdDevCombined, F_Points: trueSampsonErr, F_Points: trueSampsonErrStdDev, F_Points: trueRootSampsonErr, F_Points: trueRootSampsonErrStdDev, F_Points: quality, H_Lines: featureCountGood, H_Lines: trueFeatureCountGood, H_Lines: featureCountComplete, H_Lines: trueFeatureCountComplete, H_Lines: inlierCountOwnGood, H_Lines: trueInlierCountOwnGood, H_Lines: inlierCountOwnComplete, H_Lines: trueInlierCountOwnComplete, H_Lines: inlierCountCombined, H_Lines: trueInlierCountCombined, H_Lines: sampsonErrOwn, H_Lines: sampsonErrComplete, H_Lines: sampsonErrCombined, H_Lines: sampsonErrStdDevCombined, H_Lines: trueSampsonErr, H_Lines: trueSampsonErrStdDev, H_Lines: trueRootSampsonErr, H_Lines: trueRootSampsonErrStdDev, H_Lines: quality, H_Points: featureCountGood, H_Points: trueFeatureCountGood, H_Points: featureCountComplete, H_Points: trueFeatureCountComplete, H_Points: inlierCountOwnGood, H_Points: trueInlierCountOwnGood, H_Points: inlierCountOwnComplete, H_Points: trueInlierCountOwnComplete, H_Points: inlierCountCombined, H_Points: trueInlierCountCombined, H_Points: sampsonErrOwn, H_Points: sampsonErrComplete, H_Points: sampsonErrCombined, H_Points: sampsonErrStdDevCombined, H_Points: trueSampsonErr, H_Points: trueSampsonErrStdDev, H_Points: trueRootSampsonErr, H_Points: trueRootSampsonErrStdDev, H_Points: quality, time (min)" >> $OUT

####################### Castle ################################################## 
#"should work" examples
sh MultipleCueEstimationList.sh "castle-P30/0000.png" "castle-P30/0003.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0002.png" "castle-P30/0003.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0001.png" "castle-P30/0002.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0003.png" "castle-P30/0004.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0004.png" "castle-P30/0005.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0005.png" "castle-P30/0006.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0008.png" "castle-P30/0009.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0013.png" "castle-P30/0014.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0014.png" "castle-P30/0015.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0015.png" "castle-P30/0016.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0020.png" "castle-P30/0021.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0021.png" "castle-P30/0022.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0018.png" "castle-P30/0019.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0026.png" "castle-P30/0028.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0025.png" "castle-P30/0026.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0026.png" "castle-P30/0027.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0004.png" "castle-P30/0006.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0016.png" "castle-P30/0018.png" $OUT
#"should not work" examples
sh MultipleCueEstimationList.sh "castle-P30/0005.png" "castle-P30/0007.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0023.png" "castle-P30/0024.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0028.png" "castle-P30/0029.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0001.png" "castle-P30/0007.png" $OUT
sh MultipleCueEstimationList.sh "castle-P30/0003.png" "castle-P30/0006.png" $OUT

###################### Entry ####################################################
sh MultipleCueEstimationList.sh "" ""  $OUT
sh MultipleCueEstimationList.sh "entry-P10/0000.png" "entry-P10/0001.png" $OUT
sh MultipleCueEstimationList.sh "entry-P10/0008.png" "entry-P10/0009.png" $OUT
#"should not work" examples
sh MultipleCueEstimationList.sh "entry-P10/0007.png" "entry-P10/0009.png" $OUT
sh MultipleCueEstimationList.sh "entry-P10/0007.png" "entry-P10/0008.png" $OUT
sh MultipleCueEstimationList.sh "entry-P10/0003.png" "entry-P10/0005.png" $OUT
sh MultipleCueEstimationList.sh "entry-P10/0005.png" "entry-P10/0006.png" $OUT

###################### Fountain ####################################################

sh MultipleCueEstimationList.sh "fountain-P11/0000.png" "fountain-P11/0001.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0002.png" "fountain-P11/0003.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0004.png" "fountain-P11/0005.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0005.png" "fountain-P11/0006.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0000.png" "fountain-P11/0003.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0008.png" "fountain-P11/0009.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0009.png" "fountain-P11/0010.png" $OUT
sh MultipleCueEstimationList.sh "fountain-P11/0008.png" "fountain-P11/0010.png" $OUT

###################### HerzJesu ####################################################

sh MultipleCueEstimationList.sh "herz-jesu-P8/0000.png" "herz-jesu-P8/0001.png" $OUT
sh MultipleCueEstimationList.sh "herz-jesu-P8/0001.png" "herz-jesu-P8/0002.png" $OUT
sh MultipleCueEstimationList.sh "herz-jesu-P8/0002.png" "herz-jesu-P8/0003.png" $OUT
sh MultipleCueEstimationList.sh "herz-jesu-P8/0004.png" "herz-jesu-P8/0005.png" $OUT
sh MultipleCueEstimationList.sh "herz-jesu-P8/0006.png" "herz-jesu-P8/0007.png" $OUT
#"should not work" examples
sh MultipleCueEstimationList.sh "herz-jesu-P8/0000.png" "herz-jesu-P8/0002.png" $OUT

###################### Rest ####################################################

sh MultipleCueEstimationList.sh "brussles/rdimage.000.ppm" "brussles/rdimage.001.ppm" $OUT
sh MultipleCueEstimationList.sh "brussles/rdimage.001.ppm" "brussles/rdimage.004.ppm" $OUT

sh MultipleCueEstimationList.sh "rathaus/rdimage.001.ppm" "rathaus/rdimage.002.ppm" $OUT
#"should not work" examples
sh MultipleCueEstimationList.sh "rathaus/rdimage.002.ppm" "rathaus/rdimage.006.ppm" $OUT

sh MultipleCueEstimationList.sh "semper/rdimage.000.ppm" "semper/rdimage.002.ppm" $OUT

#sh MultipleCueEstimationList.sh "" ""  $OUT

echo " ++++++++++++ ALL DONE +++++++++++++"

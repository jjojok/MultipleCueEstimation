#!/bin/sh
if [ ! -d "Results" ]
then
	mkdir "Results"
fi

PREFIX="/home/johannes/MultipleCueEstimation/Datasets/"

OUT="Results/resultList_cameras.csv"

echo "img pair, F_result gt dist, F_result gt angle dist, F_points gt dist, F_points gt angle dist, F_hlines gt dist, F_hlines gt angle dist, F_hpoints gt dist, F_hpoints gt angle dist, Num used correspondences" >> $OUT

####################### Castle ################################################## 
#"should work" examples
sh CameraMultipleCueEstimationList.sh "castle-P30/0000.png" "castle-P30/0003.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0002.png" "castle-P30/0003.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0001.png" "castle-P30/0002.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0003.png" "castle-P30/0004.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0004.png" "castle-P30/0005.png" $OUT

###################### HerzJesu ####################################################

sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0000.png" "herz-jesu-P8/0001.png" $OUT
sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0001.png" "herz-jesu-P8/0002.png" $OUT
sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0002.png" "herz-jesu-P8/0003.png" $OUT
sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0004.png" "herz-jesu-P8/0005.png" $OUT
sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0006.png" "herz-jesu-P8/0007.png" $OUT
#"should not work" examples
sh CameraMultipleCueEstimationList.sh "herz-jesu-P8/0000.png" "herz-jesu-P8/0002.png" $OUT

sh CameraMultipleCueEstimationList.sh "castle-P30/0005.png" "castle-P30/0006.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0008.png" "castle-P30/0009.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0013.png" "castle-P30/0014.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0014.png" "castle-P30/0015.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0015.png" "castle-P30/0016.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0020.png" "castle-P30/0021.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0021.png" "castle-P30/0022.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0018.png" "castle-P30/0019.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0026.png" "castle-P30/0028.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0025.png" "castle-P30/0026.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0026.png" "castle-P30/0027.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0004.png" "castle-P30/0006.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0016.png" "castle-P30/0018.png" $OUT
#"should not work" examples
sh CameraMultipleCueEstimationList.sh "castle-P30/0005.png" "castle-P30/0007.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0023.png" "castle-P30/0024.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0028.png" "castle-P30/0029.png" $OUT
sh CameraMultipleCueEstimationList.sh "castle-P30/0003.png" "castle-P30/0006.png" $OUT

###################### Entry ####################################################
sh CameraMultipleCueEstimationList.sh "entry-P10/0000.png" "entry-P10/0001.png" $OUT
sh CameraMultipleCueEstimationList.sh "entry-P10/0008.png" "entry-P10/0009.png" $OUT
#"should not work" examples
sh CameraMultipleCueEstimationList.sh "entry-P10/0007.png" "entry-P10/0009.png" $OUT
sh CameraMultipleCueEstimationList.sh "entry-P10/0007.png" "entry-P10/0008.png" $OUT
sh CameraMultipleCueEstimationList.sh "entry-P10/0003.png" "entry-P10/0005.png" $OUT
sh CameraMultipleCueEstimationList.sh "entry-P10/0005.png" "entry-P10/0006.png" $OUT

###################### Fountain ####################################################

sh CameraMultipleCueEstimationList.sh "fountain-P11/0000.png" "fountain-P11/0001.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0002.png" "fountain-P11/0003.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0004.png" "fountain-P11/0005.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0005.png" "fountain-P11/0006.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0000.png" "fountain-P11/0003.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0008.png" "fountain-P11/0009.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0009.png" "fountain-P11/0010.png" $OUT
sh CameraMultipleCueEstimationList.sh "fountain-P11/0008.png" "fountain-P11/0010.png" $OUT

###################### Rest ####################################################

sh CameraMultipleCueEstimationList.sh "brussles/rdimage.000.ppm" "brussles/rdimage.001.ppm" $OUT
sh CameraMultipleCueEstimationList.sh "brussles/rdimage.001.ppm" "brussles/rdimage.004.ppm" $OUT

sh CameraMultipleCueEstimationList.sh "rathaus/rdimage.001.ppm" "rathaus/rdimage.002.ppm" $OUT
#"should not work" examples
sh CameraMultipleCueEstimationList.sh "rathaus/rdimage.002.ppm" "rathaus/rdimage.006.ppm" $OUT

sh CameraMultipleCueEstimationList.sh "semper/rdimage.000.ppm" "semper/rdimage.002.ppm" $OUT

#sh CameraMultipleCueEstimationList.sh "" ""  $OUT

echo " ++++++++++++ ALL DONE +++++++++++++"

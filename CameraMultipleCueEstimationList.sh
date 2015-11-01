#!/bin/bash
PREFIX="/home/johannes/MultipleCueEstimation/Datasets"

path1="$PREFIX/$1"
path2="$PREFIX/$2"

cam1="$path1.camera"
cam2="$path2.camera"

#base1=$(echo $1 | cut -d'.' -f 1-)
base1=$(echo $1 | sed -e "s/\//_/g")
#base2=$(echo $2 | cut -d'.' -f 1-)
base2=$(echo $2 | sed -e "s/\//_/g")

basedir="/home/johannes/MultipleCueEstimation/Results/Images"

if [ ! -d "$basedir" ]; then
  mkdir $basedir
fi

imgdir="/home/johannes/MultipleCueEstimation/Results/Images/$base1-$base2"

if [ ! -d "$imgdir" ]; then
  mkdir $imgdir
fi

echo "Computing Cam $1 and $2"
echo -n "Computation: "
for i in 1 2 3 4 5 6 7 8 9 10 
do
subdir="$imgdir/$i"
echo -n "$i..."

#if [ ! -f "$subdir/F_H_points.csv" ]; then
#$points="$subdir/F_H_points.csv"
#else
#$points="0"
#fi

nice -n 19 ./ComputeCameras/build/ComputeCameras "$subdir/F_gt.csv" "$imgdir/img1.camera" "$imgdir/img1.png" "$imgdir/img2.png" "$subdir/F_result.csv" "$subdir/F_points.csv" "$subdir/F_H_lines.csv" "$subdir/F_H_points.csv" $i>> $3

#subdir="$imgdir/$i"

#if [ ! -d "$imgdir/$i" ]; then
#  mkdir $subdir
#fi

#for file in $(find "/home/johannes/MultipleCueEstimation" -maxdepth 1 -type f -name "*.png")
#do	
#	mv -f $file $subdir
#done

#mv -f "F_result.csv" $subdir
#mv -f "F_points.csv" $subdir
#mv -f "F_H_points.csv" $subdir
#mv -f "F_H_lines.csv" $subdir
#mv -f "H1_points.csv" $subdir
#mv -f "H1_lines.csv" $subdir
#mv -f "H2_points.csv" $subdir
#mv -f "H2_lines.csv" $subdir
#mv -f "F_gt.csv" $subdir

done

echo ""



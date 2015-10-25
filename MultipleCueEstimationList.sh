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

echo "$(date) Computing $1 and $2"
echo -n "Computation: "
for i in 1 2 3 4 5 6 7 8 9 10 
do
echo -n "$i..."
nice -n 19 ./build/MultipleCueEstimation $path1 $path2 7 $cam1 $cam2 >> $3

subdir="$imgdir/$i"

if [ ! -d "$imgdir/$i" ]; then
  mkdir $subdir
fi

for file in $(find "/home/johannes/MultipleCueEstimation" -maxdepth 1 -type f -name "*.png")
do	
	mv -f $file $subdir
done
done

echo ""



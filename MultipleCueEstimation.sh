#path=$1
path="Datasets/entry-P10"

for file in $(find $path -type f -name '*png' | sort)
do
	echo "$file"
	if [ -f "$lastFile" ]
	then
		cam1=$(echo "$lastFile" | sed -e "s/png/png.camera/g") 
		cam2=$(echo "$file" | sed -e "s/png/png.camera/g") 
		echo "Computing: $lastFile & $file; $cam1 & $cam2"
		./build/MultipleCueEstimation $lastFile $file 7 $cam1 $cam2
  	fi
	lastFile=$file
done

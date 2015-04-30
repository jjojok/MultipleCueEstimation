path=$1
name=$2
for file in $(find $path -type f -name '*png' | sort)
do
	if [ -f "$lastFile" ]
	then
		cam1=$(echo "$lastFile" | sed -e "s/png/png.camera/g") 
		cam2=$(echo "$file" | sed -e "s/png/png.camera/g") 
		echo "Computing: $lastFile & $file; $cam1 & $cam2 ..."
		#./build/MultipleCueEstimation $lastFile $file 7 $cam1 $cam2 >> $name
  	fi
	lastFile=$file
done

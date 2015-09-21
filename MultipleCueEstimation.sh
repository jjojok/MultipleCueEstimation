path=$1
name=$2
extension=$3
search="*$extension"
for file in $(find $path -type f -name $search | sort)
do
	if [ -f "$lastFile" ]
	then
		cam1=$(echo "$lastFile" | sed -e "s/$extension/$extension.camera/g") 
		cam2=$(echo "$file" | sed -e "s/$extension/$extension.camera/g") 
		echo "Computing: $lastFile & $file; $cam1 & $cam2 for 7..."
		./build/MultipleCueEstimation $lastFile $file 7 $cam1 $cam2 >> $name
  	fi
	lastFile=$file
done

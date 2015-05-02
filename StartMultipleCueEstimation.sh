if [ ! -d "Results" ]
then
	mkdir "Results"
fi

############################ TODO: loop through all dataset folders ########################

path=$1
name=$(basename $path)

echo "Starting computation of: $path"

name="Results/$name.csv"

echo "File1,File2,RefinedFError,RefinedFInlier,RefinedFSlectedError,RefinedFSelectedInlier,RefinedFSelectedPoints,RefinedFGoodSelectedPoints,RefinedFGoodMatchedError,RefinedFGoodMatches,CombinedMatches,Estimation1,Estimatio1Error,Estimation1CombinedError,Estimation1Inlier,Estimation1SelectedError,Estimation1SelectedInlier,Estimation1GoodMatchError,Estimation2,Estimatio2Error,Estimation2CombinedError,Estimation2Inlier,Estimation2SelectedError,Estimation2SelectedInlier,Estimation2GoodMatchError,Estimation3,Estimatio3Error,Estimation3CombinedError,Estimation3Inlier,Estimation3SelectedError,Estimation3SelectedInlier,Estimation3GoodMatchError,Computationtime" >> $name

sh MultipleCueEstimation.sh $path $name

echo "done."

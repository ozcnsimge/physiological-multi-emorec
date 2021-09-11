# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
# CHOOSE THE CLASSIFIERS YOU WANT TO TRAIN
for clas in stresnetM fcnM resnetM; do
  for dataset in distracteddriving; do
    for i_fold in 00; do
      python tune_one.py "$1" "$dataset"_fold_05_"$i_fold" $clas 1
    done
  done
done

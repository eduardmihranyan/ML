## Preprocessing
All the text is made lower case, all the signs except english letters, ! ?, and signs used in smiles are removed, can't is changed to can not, 're is changed to are, ' and 's are removed

## Summary of work done
Baseline fasttext training gave 91.3% accuracy, then I increased the learning rate to 1.0 which decays during training, changed -dim parameter to 200, -wordNgrams to 3 and -epochs to 5 and reached the accuracy of 93%. This was done without preprocessing. After preprocessing the accuracy increased by ~ 2% (I forgot to fix the number)

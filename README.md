jscrf
=====

JSCRF(very tiny linear-chain CRF in JavaScript)

This program is a very tiny linear-chain CRF in JavaScript.

[USAGE]

1. Set training data with the function "setTrainingData(trainingStr)".
"trainingStr" format ... each line has to be "<observation> <label>"

2. Do learning with the function "learning()". This function will estimate weights of features.

3. Predict with the function "predictLabels(input)".This function alerts predicted labels.
"input" format ... each line is assumed to a single observation
NOTE: all observations to be predict need to be included in the training data

4. You can show the feature table, weights table, detailed prediction result by the functions "genFeatureTable()", "genWeightsTable()", "genViterbiResultTable()", respectively.
These functions show the results as the html table format. 

[LICENSE]
The BSD 2-Clause License. See LICENSE.md file.


Copyright (c) 2013, Masafumi Hamamoto
All rights reserved.

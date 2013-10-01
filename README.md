jscrf
=====

JSCRF(very tiny linear-chain CRF in JavaScript)

This program is a very tiny linear-chain CRF in JavaScript.

[USAGE]

//
// create a CRF object 
//
var crf = new CRF(); 

//
// Set training data with the method "setTrainingData(trainingStr)".
// "trainingStr" format ... each line is "<observation> <label>"
//
crf.setTrainingData(trainingStr); 

//
// Do learning with the method "learning()".
// This function will estimate weights of features.
//
crf.learning();

//
// Predict with the function "predictLabels(testStr)".
// This function alerts predicted labels.
// "testStr" format ... each line is a single observation
//
crf.predictLabels(testStr);

//
// You can show the feature table, weights table, detailed prediction 
// result by the functions "genFeatureTable()", "genWeightsTable()", 
// "genViterbiResultTable()", respectively.
// These functions show the results as the html table format. 
//
$('#feature-table').html(crf.genFeatureTable());
$('#weights-table').html(crf.genWeightsTable());
$('#viterbi-table').html(crf.genViterbiResultTable());

[LICENSE]
The BSD 2-Clause License. See LICENSE.md file.


Copyright (c) 2013, Masafumi Hamamoto
All rights reserved.

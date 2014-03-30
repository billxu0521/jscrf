jscrf
=====

JSCRF(very tiny linear-chain CRF in JavaScript)

This program is a very tiny linear-chain CRF in pure JavaScript.
Any other library is not required.

[version] 

0.1.0

[Demo]

See index.html

[Detail Usage]

//
// Create a CRF object 
//
var crf = new CRF(); 

// or, you can create an object for Node.js in the following manner;
// var crf = require("jscrf.js").createCRF();

//
// Your CRF object accepts some parameters;
// 
// epsilon: strength of each learning iteration (default: 0.01)
// maxIteration: maximum number of iteration when learning does not converge (default: 10)
// threshold: difference value for quiting learning (default: 0.0001)
//
var option = {maxIteration: 100, threshold: 0.00001};

//
// Also you are able to control which kinds of feature patterns are adopted.
// Feature patterns are currently fixed, though other non-tiny CRF programs
// use a feature template.
//
// This program prepares following eleven patterns (%x[] and %y[] denote observations and labels, respectively): 
//
// U00:%x[0] (current observation)
// U01:%x[-1] (left observation)
// U02:%x[-1]/%x[0] (left and current observation)
// U03:%x[-2] (left observation of the left)
// U04:%x[-2]/%x[-1]/%x[0] (trigram of the left two observations)
// U05:%x[1] (right observation)
// U06:%x[0]/%x[1] (current and right observation)
// U07:%x[2] (right observation of the right)
// U08:%x[0]/%x[1]/%x[2] (trigram of the right two observations)
// U09:%x[-1]/%x[0]/%x[1] (trigram around the current observation)
// T:%y[-1] (left estimated label ... use transition feature) 
//
// These features are used by default, except 'T' (my experiments show 'T' causes low precisions).
// When you change the feature patterns, please give a setting object to
// option.usefeature in the following mannar.
//

option.usefeature = {"U00": true, "U01": true, "U03": true, "U05": true} // features except "U00", "U01", "U03", and "U05" are not used

// give the option object through the method 'setOption'
crf.setOption(option);

//
// Set training data with the method "setTrainingData(trainingStr)".
//
// ["trainingStr" format]
//  trainingStr is a string value consists of one or more sequences and the boundary of each sequence is an empty line.
//  Each line in a sequence consists of two tokens 'observation' and 'label'.  'observation' and 'label' are separated by one or more white spaces.
//  NOTE: this program is currently very tiny, so two or more information in a observation cannot be used.
// 
var trainingStr = "This noun\nis verb\na article\ntest noun\n. mark\n\nThat noun\n...";
crf.setTrainingData(trainingStr); 

//
// Do learning with the method "learning()".
// This function will estimate weights of features.
//
// Training algorithm is steepest decent method. All weights are normalized so that whose L2 norm ||w|| to be one.
//
crf.learning();

//
// Predict with the function "predictLabels(testStr)".
//
// "testStr" format is almost same as "trainingStr" without any labels.
//
var testStr = "This\nis\nvery\ntiny\nprogram\n.\n\n";
var predResult = crf.predictLabels(testStr);

// predictLabels returns as an two-dimentional array; predResult[i][j] is the predicted label of the j-th observation in the i-th sequence

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


Copyright (c) 2013, 2014, Masafumi Hamamoto
All rights reserved.

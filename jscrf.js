/**
 * jscrf.js
 *
 * a Conditional Random Fields program by JavaScript
 *
 * @version 0.0.1-alpha
 * @author Masafumi Hamamoto
 *
 */

var weights = {};
var epsilon = 0.1; // learning

// phi is the feature function: phi[f] = 1 (exist) of 0 (not exist)
var phi = {};

var bosObs = "BOS";
var eosObs = "EOS";
var bosLabel = "__BOS__";
var eosLabel = "__EOS__";

var learningDataSet = new Array();
var labelList = {};
var labelNum = 0;

var lastPredResult = {}; // for genViterbiResultTable


///////////////////////// prediction /////////////////////////////

/**
 * predict input sequence represented by string
 * input string must be separated by \n and each line is assumed as an observation
 */
function predictLabels(input)
{
  var lines = input.split(/\n+/);
	var line;
	var i;
	var data;
	var seqStr = "";
	var x = new Array(); // observation
	
	for(i = 0; i < lines.length; i++) {
		line = lines[i];
		line = line.replace(/^\s+/, "");
		if(line == ""){
			 continue;
		}else if(line == "EOS"){
			 break;
		}
		x.push(line);
	}

	var estimatedSeq = predict(x);
	for(i = 0; i < x.length; i++){
		seqStr += estimatedSeq[i] + ":";
	}
	
	lastPredResult.xlength = x.length;
	lastPredResult.estimated = estimatedSeq;
	
	alert(seqStr);
}

/**
 * predict sequence
 */
function predict(x) {
	var bosNode, eosNode;
	var node;
	var estimatedSeq;
	
	// make lattice
	var latticeInfo = makeLattice(x);
	bosNode = latticeInfo["bosNode"];
	eosNode = latticeInfo["eosNode"];
	lastPredResult.bosNode = bosNode;
	lastPredResult.eosNode = eosNode;

	// viterbi
	viterbi(bosNode);
	
	// backtrack
	estimatedSeq = new Array();
	for(node = eosNode.maxLeftNode; node != bosNode; node = node.maxLeftNode){
		estimatedSeq.push(node.y);
	}
	estimatedSeq = estimatedSeq.reverse();
	
	return estimatedSeq;
}

/**
 * make lattice to estimate
 */
function makeLattice(x) { 
	var i;
	var lastNode, newNode, leftNode;
	var bosNode = makeNode(bosObs, bosLabel, null);
	var leftNode = bosNode;
	
	for (i = 0; i < x.length; i++) {
		lastNode = null;
		for (label in labelList) {
			newNode = makeNode(x[i], label, leftNode);
			newNode.next = lastNode;
			lastNode = newNode;
		}
		leftNode = lastNode;
	}
	eosNode = makeNode(eosObs, eosLabel, leftNode);
	
	return {bosNode: bosNode, eosNode: eosNode};
}


/**
 * do viterbi algorithm: connect from each node to the best left node
 */
function viterbi(bosNode) {
	var baseNode, node, leftNode;
	var innerProd, score, maxScore;
	var transFeature, obsFeature;
	
	bosNode.score = 0;
	for (baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right) { // move position
		for (node = baseNode; node != null; node = node.next) { // check each label at baseNode position
			maxScore = -100000000;
			
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // check all left nodes' score
				transFeature = node.y + ":" + leftNode.y;
				obsFeature = node.x + "#" + node.y;
				innerProd = weights[transFeature]*phi[transFeature] + weights[obsFeature]*phi[obsFeature];
				score = innerProd + leftNode.score;
				if(maxScore < score){
					maxScore = score;
					node.score = score;
					node.maxLeftNode = leftNode;
				}
			}
		}
	}
}


///////////////////////// DATA READING ////////////////////////////

/**
 * parse a string of training data
 */
function setTrainingData(trainingStr) {
	var lines = trainingStr.split(/\n+/);
	var line;
	var newNode;
	var lastNode;
	var x, y;
	var tuple;
	var lineCount;
	var bosNode;
	var eosNode;

	var isBOS = true;
	for(lineCount = 0; lineCount < lines.length; lineCount++){
		line = lines[lineCount].replace(/^\s+/, "").replace(/\s+$/, "");
		
		if(line == ""){
			// skip
			
		} else if (line == "EOS") {
			eosNode = makeNode(eosObs, eosLabel, lastNode);
			learningDataSet.push({
				bosNode: bosNode,
				eosNode: eosNode
			});
			
			 // set EOS transition and observation feature
			phi[eosNode.y + ":" + eosNode.left.y] = 1;
			phi[eosObs + "#" + eosLabel] = 1;

			isBOS = true;
			
		} else {
			if (isBOS) { // init bosNode
				bosNode = makeNode(bosObs, bosLabel, null);
				lastNode = bosNode;
				isBOS = false;
			}
			
			tuple = line.split(/\s+/);
			x = tuple[0];
			y = tuple[1];
			newNode = makeNode(x, y, lastNode);
			lastNode = newNode;
		
			if (!labelList[y]) {
				labelList[y] = 1;
				labelNum++;
			}
		
			phi[newNode.y + ":" + newNode.left.y] = 1; // transition feature
			phi[newNode.x + "#" + newNode.y] = 1; // observation feature
		}
	}
	
	addNodeChain();

	// init weights
	for (feature in phi) {
		weights[feature] = 0;
	}
}

/**
 * add dummy nodes for alpha and beta calcuration
 */
function addNodeChain() {
	var node;
	var addLabelList;
	var i;
	var label;
	var newNode;
	var lastNode;
	var dataIdx;
	var bosNode;
	var eosNode;
	
	for(dataIdx = 0; dataIdx < learningDataSet.length; dataIdx++){
		bosNode = learningDataSet[dataIdx]["bosNode"];
		eosNode = learningDataSet[dataIdx]["eosNode"];

		for (node = bosNode.right; node != eosNode; node = node.right) {
			// get label list except node.y
			addLabelList = new Array();
			for (label in labelList) {
				if(label != node.y){
					addLabelList.push(label);
				}
			}
		
			// add nextNode chain
			lastNode = node;
			for (i = 0; i < addLabelList.length; i++){
				label = addLabelList[i];
				newNode = makeNode(node.x, label);
				newNode.left = node.left;
				newNode.right = node.right;
				lastNode.next = newNode;
			
				lastNode = newNode;
				
				// add dummy feature
				phi[newNode.y + ":" + newNode.left.y] = 1; // transition feature
				phi[newNode.x + "#" + newNode.y] = 1; // observation feature

			}
		}
	}
}

/////////////////////////// LEARNING ///////////////////////////////

/**
 * do learning weights
 */
function learning() {
	var node;
	var feature;
	var bosNode;
	var eosNode;
	var deltaL = {}; // target function
	var maxIte = 10; // limit of iteration
	var oldWeights;

	var iteCount;
	for (iteCount = 1; iteCount <= maxIte; iteCount++) {
		oldWeights = weights;
		
		for (feature in phi) {
			deltaL[feature] = 0;
		}
		
		for (dataIdx = 0; dataIdx < learningDataSet.length; dataIdx++){
			bosNode = learningDataSet[dataIdx]["bosNode"];
			eosNode = learningDataSet[dataIdx]["eosNode"];
	
			// calc all alpha and beta
			calcAlpha(eosNode);
			calcBeta(bosNode);
	
			// calc Z
			var Z = 0;
			for (node = eosNode.left; node != null; node = node.next) {
				Z += node.alpha;
			}

			// calc Sum_{Y}(P(Y|X)*phi(X, Y))
			var allProbPhi = calcAllProbPhi(bosNode, Z);

			// calc phi(X, Y)
			// NOTE: Y is a label sequence in the training data
			var allPhi = calcAllPhi(bosNode);

			// delta_{w}L(w^{old})
			for (feature in phi) {
				deltaL[feature] += allPhi[feature] - allProbPhi[feature];
			}
		}
		
		// update new weights
		for (feature in phi) {
			weights[feature] = oldWeights[feature] + epsilon*deltaL[feature];
		}

		// chech converged
		// TODO
	}

	// test
//	for (feature in phi) {
//		alert("feature: " + feature + ", P(Y|X)*phi(X, Y)= " + allProbPhi[feature] + " , deltaL(w): " + deltaL[feature] + ", weights: " + weights[feature]);
//	}

	
//	printInfo(bosNode); // TEST
}


/**
 * calc Sum_{Y}(P(Y|X)*phi(X, Y))
 */
function calcAllProbPhi(bosNode, Z) {
	var node;
	var baseNode;
	var psi, alpha, beta, prob;
	var transFeature, observFeature;
	var allProbPhi = {};

	for (feature in phi) {
		allProbPhi[feature] = 0;
	}

	// calc conditional merginal probability  P(y_t, y_{t-1}|X)
	for(baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right){ // move t
		for (node = baseNode; node != null; node = node.next) { // change t's label
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // all left (change (t-1)'s label)
				psi = calcPsi(node.y, leftNode.y, node);
				prob = psi * leftNode.alpha * node.beta / Z;
				
				// add prob*phi to allProb
				// corresponding to each feature
				transFeature = node.y + ":" + leftNode.y;
				observFeature = node.x + "#" + node.y;

				allProbPhi[transFeature] += prob*phi[transFeature];
				allProbPhi[observFeature] += prob*phi[observFeature];
			}
		}
	}
	
	return allProbPhi;
}

/**
 * calc phi(X, Y) = Sum_{t}(phi(X, y_t, y_{t-1})
 * NOTE: Y is a label sequence of the training data
 */
function calcAllPhi(bosNode) {
	var feature;
	var node, baseNode;
	var allPhi = {};
	var transFeature, observFeature;
	
	for (feature in phi) {
		allPhi[feature] = 0;
	}
	for (node = bosNode.right; node != null; node = node.right) {
		transFeature = node.y + ":" + node.left.y;
		observFeature = node.x + "#" + node.y;
		allPhi[transFeature] += phi[transFeature];
		allPhi[observFeature] += phi[observFeature];
	}
	
	return allPhi;
} 


/////////////////////// NODE FUNCTIONS ////////////////////////

/**
 * make a new node
 */
function makeNode(x, y, leftNode) {
	var node = {};
	
	node.x = x;
	node.y = y;
	node.left = leftNode;
	node.right = null;
	node.alpha = 0;
	node.beta = 0;
	node.maxLeftNode = null;
	
	node.next = null; // pointer to a node whose x is same and y is not same
	if(leftNode){
		leftNode.right = node;
	}

	return node;
}


/**
 * calc given node's alpha value
 */
function calcAlpha(node) {
	var leftNode;
	node.alpha = 0;
	
	if(node.x == bosObs && node.y == bosLabel){
		node.alpha = 1;
	}else{
		// sum all psi(label,leftLabel)*leftAlpha(leftLabel)
		for(leftNode = node.left; leftNode != null; leftNode = leftNode.next){
			if(leftNode.alpha == 0){
				calcAlpha(leftNode);
			}
			
			var psi = calcPsi(node.y, leftNode.y, node);
			node.alpha += psi*leftNode.alpha;
		}
	}
}

/**
 * calc given node's beta value
 */
function calcBeta(node) {
	var rightNode;
	node.beta = 0;

	if(node.x == eosObs && node.y == eosLabel){
		node.beta = 1;
	}else{
		// sum all psi(label,rightLabel)*rightBeta(rightLabel)
		for(rightNode = node.right; rightNode != null; rightNode = rightNode.next){
			if(rightNode.beta == 0){
				calcBeta(rightNode);
			}
			var psi = calcPsi(rightNode.y, node.y, node.right);
			node.beta += psi*rightNode.beta;
		}
	}
}

/**
 * psi(y_t, y_{t-1}, X, w) = Math.exp(w * phi(X, y_t, y_{t-1}))
 */ 
function calcPsi(label, leftLabel, node)
{
	var transFeature = label + ":" + leftLabel;
	var observFeature = node.x + "#" + node.y;
	var prod = 0; // product
	
	// w * phi(X, y_t, y_t-1)
	if (phi[transFeature] == 1) {
		prod += weights[transFeature];
	}
	if (phi[observFeature] == 1) {
		prod += weights[observFeature];
	}
	
	return Math.exp(prod);
}


///////////////////////// presentation //////////////////////////////

/**
 * test: show all features as a table
 */
function genFeatureTable() {
	var html = "<table border='2'>";
	
	html += "<tr><th>feature</th></tr>";
	for (feature in phi) {
		html += "<tr><td>" + feature + "</td></tr>";
	}
	html += "</table>";

	return html;
}

/**
 * show all feature-weight map as a table
 */
function genWeightsTable() {
	var html = '<table class="weight-table" border="2">';
	var feature;
	
	html += '<tr><th>feature</th><th>value</th></tr>';
	for (feature in phi){
		html += "<tr><td>" + feature + "</td><td>" + weights[feature] + "</td></tr>";
	}
	html += "</table>";
	
	return html;
}

/**
 * show viterbi algorithm result as a table
 */
function genViterbiResultTable() {
	var html = '<table class="viterbi-table" border="2">';
	var node;
	var baseNode;

	// convert lattice to 2D array
	var colsArray = new Array(lastPredResult.xlength+2);
	var colIdx = 0, nodeIdx = 0;

	for (baseNode = lastPredResult.bosNode; baseNode != null; baseNode = baseNode.right, colIdx++) {
		colsArray[colIdx] = new Array();
		for (node = baseNode; node != null; node = node.next) {
			colsArray[colIdx].push(node);
		}
	}

	// header
	html += "<tr>";
	for (colIdx = 0; colIdx < colsArray.length; colIdx++) {
		html += "<th>" + colsArray[colIdx][0].x + "</th>";
	}
	html += "</tr>";

	// value
	for (nodeIdx = labelNum-1; 0 <= nodeIdx; nodeIdx--) {
		html += "<tr>";
		if(nodeIdx == labelNum-1){
			node = colsArray[0][0];
			html += genNodeInfoTD(node, true);
		}
		for (colIdx = 1; colIdx < colsArray.length-1; colIdx++) {
			node = colsArray[colIdx][nodeIdx];
			html += genNodeInfoTD(node);
		}
		if(nodeIdx == labelNum-1){
			node = colsArray[colsArray.length-1][0];
			html += genNodeInfoTD(node, true);
		}
		html += "</tr>";
	}
	
	html += "</table>";
	
	return html;
}

function genNodeInfoTD(node, doRowSpan) {
	var td;

	if(doRowSpan){
		td = '<td rowspan="' + labelNum + '">';
	}else{
		td = "<td>";
	}
	td += '<table border="1">';
	td += "<tr><td>label</td><td>" + node.y + "</td></tr>";
	td += "<tr><td>score</td><td>" + node.score + "</td></tr>";
	if(node.maxLeftNode){
		td += "<tr><td>leftLabel</td><td>" + node.maxLeftNode.y + "</td></tr>";
	}
	
	td += "</table>";
	
	td += "</td>";
	return td;
}

////////////////////////// FOR DEBUG ////////////////////////////////

/*
 * FOR DEBUG: print answer of problems in the book
 */
function printInfo(bosNode) {
	var node;
	var baseNode;
	var node2, node3; // node of x2 and x3

	for (baseNode = bosNode; baseNode != null; baseNode = baseNode.right){
		for (node = baseNode; node != null; node = node.next) {
			alert("node.x: " + node.x + ", node.y=: " + node.y + ", alpha: "+node.alpha+", beta: "+node.beta);
		}
	}

	alert("Z = " + Z);
	
	// P(y3, y2|X)
	for(node3 = bosNode.right.right.right; node3 != null; node3 = node3.next){
		for(node2 = node3.left; node2 != null; node2 = node2.next){
			var psi = calcPsi(node3.y, node2.y, node3);
			var alpha = node2.alpha;
			var beta = node3.beta;
			var prob = psi*alpha*beta/Z;
			alert("P("+node3.y+", "+node2.y+"|X) = " + prob + " = " + psi + " * " + alpha + " * " + beta + " / " + Z);
		}
	}
}

/**
 * return phi in the probrem 5.1 of the book
 */ 
function calcPsiTest(label, leftLabel, node)
{

	if (leftLabel == bosLabel) { // psi_1
		return 1.0;
	}else if (node.x == "x2") {
		if(label == "c1"){
			if(leftLabel == "c1"){
				return 0.2;
			}else if (leftLabel == "c2") {
				return 0.3;
			}
		}else {
			return 0.1;
		}
	}else if (node.x == "x3") {
		if (label == "c1") {
			return 0.2;
		}else {
			return 0.1;
		}
	}else if (node.x == "x4") {
		if(label == "c1" && leftLabel == "c1"){
			return 0.3;
		}else if(label == "c1" && leftLabel == "c2"){
			return 0.1;
		}else if(label == "c2" && leftLabel == "c1"){
			return 0.2;
		}else if(label == "c2" && leftLabel == "c2"){
			return 0.1;
		}
	}else if (label == eosLabel){
		return 1.0;
	}
}


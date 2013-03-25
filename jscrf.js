/**
 * jscrf.js
 *
 * a Conditional Random Fields program by JavaScript
 *
 * @version 0.0.1
 * @author Masafumi Hamamoto
 *
 */

/////////////////////// groval variables  /////////////////////////////

var __weights;
var __epsilon = 0.1; // learning weight

// __phi is the feature function: __phi[f] = 1 (exist) of 0 (not exist)
var __phi = {};
var __id2feature = new Array();

var __bosObs = "__BOS__";
var __eosObs = "__EOS___";
var __bosLabel = "__BOSLABEL__";
var __eosLabel = "__EOSLABEL__";

var __learningDataSet = new Array();
var __labelList = {};
var __labelNum = 0;
var __featureNum = 0;

var __lastPredResult = {}; // for genViterbiResultTable


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
	__lastPredResult.xlength = x.length;
	__lastPredResult.estimated = estimatedSeq;

	return estimatedSeq;
	
	/*
	for(i = 0; i < x.length; i++){
		seqStr += estimatedSeq[i] + ":";
	}
	seqStr = seqStr.substr(0, seqStr.length-1);
	
	return seqStr;
	*/
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
	__lastPredResult.bosNode = bosNode;
	__lastPredResult.eosNode = eosNode;

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
	var bosNode = makeNode(__bosObs, __bosLabel, null);
	var leftNode = bosNode;
	
	for (i = 0; i < x.length; i++) {
		lastNode = null;
		for (label in __labelList) {
			newNode = makeNode(x[i], label, leftNode);
			newNode.next = lastNode;
			lastNode = newNode;
		}
		leftNode = lastNode;
	}
	eosNode = makeNode(__eosObs, __eosLabel, leftNode);
	
	return {bosNode: bosNode, eosNode: eosNode};
}


/**
 * do viterbi algorithm: connect from each node to the best left node
 */
function viterbi(bosNode) {
	var baseNode, node, leftNode;
	var innerProd, score, maxScore;
	var transFeature, obsFeature;
	var feature;
	
	bosNode.score = 0;
	for (baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right) { // move position
		for (node = baseNode; node != null; node = node.next) { // check each label at baseNode position
			maxScore = -100000000;
			
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // check all left nodes' score
				
				innerProd = 0;
				for (feature in getFeature(node, leftNode)) {
					innerProd += __weights[feature];
				}
				
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
			eosNode = makeNode(__eosObs, __eosLabel, lastNode);
			__learningDataSet.push({
				bosNode: bosNode,
				eosNode: eosNode
			});
			
			setFeature(eosNode);

			isBOS = true;
			
		} else {
			if (isBOS) { // init bosNode
				bosNode = makeNode(__bosObs, __bosLabel, null);
				lastNode = bosNode;
				isBOS = false;
			}
			
			tuple = line.split(/\s+/);
			x = tuple[0];
			y = tuple[1];
			newNode = makeNode(x, y, lastNode);
			lastNode = newNode;
		
			if (!__labelList[y]) {
				__labelList[y] = 1;
				__labelNum++;
			}
			
			setFeature(newNode);
		}
	}
	
	addNodeChain();

	initWeights();

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
	
	for(dataIdx = 0; dataIdx < __learningDataSet.length; dataIdx++){
		bosNode = __learningDataSet[dataIdx]["bosNode"];
		eosNode = __learningDataSet[dataIdx]["eosNode"];

		for (node = bosNode.right; node != eosNode; node = node.right) {
			// get label list except node.y
			addLabelList = new Array();
			for (label in __labelList) {
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
				
				setFeature(newNode); // add dummy feature
			}
		}
	}
}

/////////////////////////// FEATURE ///////////////////////////////

/**
 * initialize weight vector
 */
function initWeights() {
	var feature;
	__weights = new Array(__featureNum);
	for (feature = 0; feature < __featureNum; feature++) {
		__weights[feature] = 0;
	}
}

/**
 * set features of given data to the feature table 
 */
function setFeature (node) {
	if(!__phi[node.y + ":" + node.left.y]){ // new transition feature
		__phi[node.y + ":" + node.left.y] = __featureNum;
		__id2feature.push(node.y + ":" + node.left.y);
		__featureNum++;
	}
	
	if(!__phi[node.x + "#" + node.y]){ // new observation feature
		__phi[node.x + "#" + node.y] = __featureNum;
		__id2feature.push(node.x + "#" + node.y);
		__featureNum++;
	}
}

/**
 * get feature ID of given node
 */
function getFeature (node, leftNode) {
	var result = {};
	
	if(__phi[node.y + ":" + leftNode.y]){
		result[__phi[node.y + ":" + leftNode.y]] = 1;
	}
	if (__phi[node.x + "#" + node.y]) {
		result[__phi[node.x + "#" + node.y]] = 1;
	}

	return result;
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
	var deltaL = new Array(__featureNum); // target function
	var maxIte = 10; // limit of iteration
	var oldWeights;

	var iteCount;
	for (iteCount = 1; iteCount <= maxIte; iteCount++) {
		oldWeights = __weights;
		
		for (feature = 0; feature < __featureNum; feature++) {
			deltaL[feature] = 0;
		}
		
		for (dataIdx = 0; dataIdx < __learningDataSet.length; dataIdx++){
			bosNode = __learningDataSet[dataIdx]["bosNode"];
			eosNode = __learningDataSet[dataIdx]["eosNode"];
	
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
			for (feature = 0; feature < __featureNum; feature++) {
				deltaL[feature] += allPhi[feature] - allProbPhi[feature];
			}
		}
		
		// update new weights
		for (feature = 0; feature < __featureNum; feature++) {
			__weights[feature] = oldWeights[feature] + __epsilon*deltaL[feature];
		}

		// chech converged
		// TODO
	}

	// test
//	for (feature = 0; feature < __featureNum; feature++) {
//		alert("feature: " + feature + ", P(Y|X)*phi(X, Y)= " + allProbPhi[feature] + " , deltaL(w): " + deltaL[feature] + ", weights: " + __weights[feature]);
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
	var feature;
	var allProbPhi = new Array();

	for (feature = 0; feature < __featureNum; feature++){
		allProbPhi[feature] = 0;
	}

	// calc conditional merginal probability  P(y_t, y_{t-1}|X)
	for(baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right){ // move t
		for (node = baseNode; node != null; node = node.next) { // change t's label
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // all left (change (t-1)'s label)

				psi = calcPsi(node, leftNode);

				prob = psi * leftNode.alpha * node.beta / Z;
				
				// add prob*phi to allProb
				// corresponding to each feature
				for (feature in getFeature(node, leftNode)) {
					allProbPhi[feature] += prob;
				}
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
	var allPhi = new Array(__featureNum);
	var transFeature, observFeature;
	
	for (feature = 0; feature < __featureNum; feature++) {
		allPhi[feature] = 0;
	}
	for (node = bosNode.right; node != null; node = node.right) {
		for (feature in getFeature(node, node.left)) {
			allPhi[feature] += 1;
		}
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
	
	if(node.x == __bosObs && node.y == __bosLabel){
		node.alpha = 1;
	}else{
		// sum all psi(label,leftLabel)*leftAlpha(leftLabel)
		for(leftNode = node.left; leftNode != null; leftNode = leftNode.next){
			if(leftNode.alpha == 0){
				calcAlpha(leftNode);
			}
			
			var psi = calcPsi(node, leftNode);

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

	if(node.x == __eosObs && node.y == __eosLabel){
		node.beta = 1;
	}else{
		// sum all psi(label,rightLabel)*rightBeta(rightLabel)
		for(rightNode = node.right; rightNode != null; rightNode = rightNode.next){
			if(rightNode.beta == 0){
				calcBeta(rightNode);
			}
			var psi = calcPsi(rightNode, node);
			node.beta += psi*rightNode.beta;
		}
	}
}

/**
 * psi(y_t, y_{t-1}, X, w) = Math.exp(w * phi(X, y_t, y_{t-1}))
 */ 
function calcPsi(node, leftNode)
{
	var prod = 0; // product
	
	// w * phi(X, y_t, y_t-1)
	for (feature in getFeature(node, leftNode)) {
		prod += __weights[feature];
	}
	
	return Math.exp(prod);
}


///////////////////////// presentation //////////////////////////////

/**
 * test: show all features as a table
 */
function genFeatureTable() {
	var html = "<table border='2'>";
	var feature;
	
	html += "<tr><th>feature</th></tr>";
	for (feature = 0; feature < __featureNum; feature++) {
		html += "<tr><td>" + __id2feature[feature] + "</td></tr>";
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
	for (feature = 0; feature < __featureNum; feature++){
		html += "<tr><td>" + __id2feature[feature] + "</td><td>" + __weights[feature] + "</td></tr>";
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
	var colsArray = new Array(__lastPredResult.xlength+2);
	var colIdx = 0, nodeIdx = 0;

	for (baseNode = __lastPredResult.bosNode; baseNode != null; baseNode = baseNode.right, colIdx++) {
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
	for (nodeIdx = __labelNum-1; 0 <= nodeIdx; nodeIdx--) {
		html += "<tr>";
		if(nodeIdx == __labelNum-1){
			node = colsArray[0][0];
			html += genNodeInfoTD(node, true);
		}
		for (colIdx = 1; colIdx < colsArray.length-1; colIdx++) {
			node = colsArray[colIdx][nodeIdx];
			html += genNodeInfoTD(node);
		}
		if(nodeIdx == __labelNum-1){
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
		td = '<td rowspan="' + __labelNum + '">';
	}else{
		td = "<td>";
	}
	td += '<table border="1">';
	td += "<tr><td>label</td><td>" + node.y + "</td></tr>";
	td += "<tr><td>score</td><td>" + node.score + "</td></tr>";
	if(node.maxLeftNode){
		td += "<tr><td>BestLeftLabel</td><td>" + node.maxLeftNode.y + "</td></tr>";
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
			var psi = calcPsi(node3, node2);

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

	if (leftLabel == __bosLabel) { // psi_1
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
	}else if (label == __eosLabel){
		return 1.0;
	}
}


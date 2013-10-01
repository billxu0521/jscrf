/**
 * jscrf.js
 *
 * a Conditional Random Fields program by JavaScript
 *
 * @version 0.0.2
 * @author Masafumi Hamamoto
 *
 */

/////////////////////// constructor ///////////////////////////
"use strict";

function CRF(option)
{
  var self = this instanceof CRF
           ? this
           : Object.create(CRF.prototype);
  option = option || {};

  self.weights = [];
  // phi is the feature function: crf.phi[f] = 1 (exist) of 0 (not exist)
  self.phi = {};
  self.id2feature = [];
  self.learningDataSet = [];
  self.labelList = {};
  self.lastPredResult = {}; // for genViterbiResultTable
  self.labelNum = 0;
  self.featureNum = 0;

  // learning weight
  if(option.epsilon){
    if(typeof option.epsilon !== "number" || option.epsilon <= 0){
      throw "invalid option.epsilon";
    }else{
      self.epsilon = option.epsilon;
    }
  }else{
    self.epsilon = 0.1;
  }

  return self;
}

///////////////////////// prediction /////////////////////////////

/**
 * predict input sequence represented by string
 * input string must be separated by \n and each line is assumed as an observation
 */
CRF.prototype.predictLabels = function(input) {
  var lines = input.split(/\n+/);
	var line;
	var i;
	var data;
	var seqStr = "";
	var x = []; // observation
	
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

	var estimatedSeq = this.predict(x);
	this.lastPredResult.xlength = x.length;
	this.lastPredResult.estimated = estimatedSeq;

	return estimatedSeq;
}

/**
 * predict sequence
 */
CRF.prototype.predict = function(x) {
	var bosNode, eosNode;
	var node;
	var estimatedSeq;
	
	// make lattice
	var latticeInfo = this.makeLattice(x);
	bosNode = latticeInfo.bosNode;
	eosNode = latticeInfo.eosNode;
	this.lastPredResult.bosNode = bosNode;
	this.lastPredResult.eosNode = eosNode;

	// viterbi
	this.viterbi(bosNode);
	
	// backtrack
	estimatedSeq = [];
	for(node = eosNode.maxLeftNode; node != bosNode; node = node.maxLeftNode){
		estimatedSeq.push(node.y);
	}
	estimatedSeq = estimatedSeq.reverse();
	
	return estimatedSeq;
}

/**
 * make lattice to estimate
 */
CRF.prototype.makeLattice = function(x) { 
	var i;
	var lastNode, newNode, leftNode;
	var bosNode = this.createBOSNode();
	var leftNode = bosNode;
  var label;
  var eosNode;
	
	for (i = 0; i < x.length; i++) {
		lastNode = null;
		for (label in this.labelList) {
			newNode = new CRFNode(x[i], label, leftNode);
			newNode.next = lastNode;
			lastNode = newNode;
		}
		leftNode = lastNode;
	}
	eosNode = this.createEOSNode(leftNode);
	
	return {bosNode: bosNode, eosNode: eosNode};
}


/**
 * do viterbi algorithm: connect from each node to the best left node
 */
CRF.prototype.viterbi = function(bosNode) {
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
				for (feature in this.getFeature(node, leftNode)) {
					innerProd += this.weights[feature];
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
CRF.prototype.setTrainingData = function(trainingStr) {
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
			eosNode = this.createEOSNode(lastNode);
			this.learningDataSet.push({
				bosNode: bosNode,
				eosNode: eosNode
			});
			
			this.setFeature(eosNode);

			isBOS = true;
			
		} else {
			if (isBOS) { // init bosNode
				bosNode = this.createBOSNode();
				lastNode = bosNode;
				isBOS = false;
			}
			
			tuple = line.split(/\s+/);
			x = tuple[0];
			y = tuple[1];
			newNode = new CRFNode(x, y, lastNode);
			lastNode = newNode;
		
			if (!this.labelList[y]) {
				this.labelList[y] = 1;
				this.labelNum++;
			}
			
			this.setFeature(newNode);
		}
	}
	
	this.addNodeChain();

	this.initWeights();
}

/**
 * add dummy nodes for alpha and beta calcuration
 */
CRF.prototype.addNodeChain = function () {
	var node;
	var addLabelList;
	var i;
	var label;
	var newNode;
	var lastNode;
	var dataIdx;
	var bosNode;
	var eosNode;

	
	for(dataIdx = 0; dataIdx < this.learningDataSet.length; dataIdx++){
		bosNode = this.learningDataSet[dataIdx].bosNode;
		eosNode = this.learningDataSet[dataIdx].eosNode;

		for (node = bosNode.right; node != eosNode; node = node.right) {
			// get label list except node.y
			addLabelList = [];
			for (label in this.labelList) {
				if(label != node.y){
					addLabelList.push(label);
				}
			}
		
			// add nextNode chain
			lastNode = node;
			for (i = 0; i < addLabelList.length; i++){
				label = addLabelList[i];
				newNode = new CRFNode(node.x, label, null);
				newNode.left = node.left;
				newNode.right = node.right;
				lastNode.next = newNode;
			
				lastNode = newNode;
				
				this.setFeature(newNode); // add dummy feature
			}
		}
		
		// add features between eosNode and lefts
		this.addEOSFeature(eosNode);
	}
}

/////////////////////////// FEATURE ///////////////////////////////

/**
 * initialize weight vector
 */
CRF.prototype.initWeights = function() {
	var feature;
	this.weights = [];
	for (feature = 0; feature < this.featureNum; feature++) {
		this.weights[feature] = 0;
	}
}

/**
 * set features of given data to the feature table 
 */
CRF.prototype.setFeature = function(node) {
	if(!this.phi[node.y + ":" + node.left.y]){ // new transition feature
		this.phi[node.y + ":" + node.left.y] = this.featureNum;
		this.id2feature.push(node.y + ":" + node.left.y);
		this.featureNum++;
	}
	
	if(!this.phi[node.x + "#" + node.y]){ // new observation feature
		this.phi[node.x + "#" + node.y] = this.featureNum;
		this.id2feature.push(node.x + "#" + node.y);
		this.featureNum++;
	}
}

/**
 * add transition features between EOS and lefts of EOS
 */
CRF.prototype.addEOSFeature = function(eosNode) {
	var leftNode;
	for(leftNode = eosNode.left; leftNode != null; leftNode = leftNode.next){
		if(!this.phi[eosNode.y + ":" + leftNode.y]){
			this.phi[eosNode.y + ":" + leftNode.y] = this.featureNum;
			this.id2feature.push(eosNode.y + ":" + leftNode.y);
			this.featureNum++;
		}
	}
}


/**
 * get feature ID of given node
 */
CRF.prototype.getFeature = function(node, leftNode) {
	var result = {};
	
	if(this.phi[node.y + ":" + leftNode.y]){
		result[this.phi[node.y + ":" + leftNode.y]] = 1;
	}
	if (this.phi[node.x + "#" + node.y]) {
		result[this.phi[node.x + "#" + node.y]] = 1;
	}

	return result;
}


/////////////////////////// LEARNING ///////////////////////////////

/**
 * do learning weights
 */
CRF.prototype.learning = function() {
	var node;
	var feature;
	var bosNode;
	var eosNode;
  var dataIdx;
	var deltaL = []; // target function
	var maxIte = 1000; // limit of iteration
	var oldWeights;
	var threshold = 0.001;

	var iteCount;
	for (iteCount = 1; iteCount <= maxIte; iteCount++) {
		oldWeights = []
		
		for (feature = 0; feature < this.featureNum; feature++) {
			deltaL[feature] = 0;
			oldWeights[feature] = this.weights[feature];
		}
		
		for (dataIdx = 0; dataIdx < this.learningDataSet.length; dataIdx++){
			bosNode = this.learningDataSet[dataIdx].bosNode;
			eosNode = this.learningDataSet[dataIdx].eosNode;
	
			// calc all alpha and beta
			eosNode.calcAlpha(this);
			bosNode.calcBeta(this);
	
			// calc Z
			var Z = 0;
			for (node = eosNode.left; node != null; node = node.next) {
				Z += node.alpha;
			}

			// calc Sum_{Y}(P(Y|X)*phi(X, Y))
			var allProbPhi = this.calcAllProbPhi(bosNode, Z);

			// calc phi(X, Y)
			// NOTE: Y is a label sequence in the training data
			var allPhi = this.calcAllPhi(bosNode);

			// delta_{w}L(w^{old})
			for (feature = 0; feature < this.featureNum; feature++) {
				deltaL[feature] += allPhi[feature] - allProbPhi[feature];
			}
		}
		
		// update new weights
		for (feature = 0; feature < this.featureNum; feature++) {
			this.weights[feature] = oldWeights[feature] + this.epsilon*deltaL[feature];
		}

		// check converged
		if(this.diffnorm(oldWeights) < threshold){
//			alert("final diff norm: " + diffnorm(oldWeights) + ", iteration: " + iteCount);
			break;
		}
	}
}

CRF.prototype.diffnorm = function(oldWeights) {
	var featureID;
	var diff;
	var norm = 0;
	
	for (featureID = 0; featureID < this.featureNum; featureID++) {
		diff = this.weights[featureID] - oldWeights[featureID];
		norm += diff*diff;
	}

	return norm;
}

/**
 * calc Sum_{Y}(P(Y|X)*phi(X, Y))
 */
CRF.prototype.calcAllProbPhi = function(bosNode, Z) {
	var node;
	var baseNode, leftNode;
	var psi, alpha, beta, prob;
	var feature;
	var allProbPhi = [];

	for (feature = 0; feature < this.featureNum; feature++){
		allProbPhi[feature] = 0;
	}

	// calc conditional merginal probability  P(y_t, y_{t-1}|X)
	for(baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right){ // move t
		for (node = baseNode; node != null; node = node.next) { // change t's label
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // all left (change (t-1)'s label)

				psi = this.calcPsi(node, leftNode);

				prob = psi * leftNode.alpha * node.beta / Z;
				
				// add prob*phi to allProb
				// corresponding to each feature
				for (feature in this.getFeature(node, leftNode)) {
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
CRF.prototype.calcAllPhi = function(bosNode) {
	var feature;
	var node, baseNode;
	var allPhi = [];
	var transFeature, observFeature;
	
	for (feature = 0; feature < this.featureNum; feature++) {
		allPhi[feature] = 0;
	}
	for (node = bosNode.right; node != null; node = node.right) {
		for (feature in this.getFeature(node, node.left)) {
			allPhi[feature] += 1;
		}
	}
	
	return allPhi;
} 

/**
 * psi(y_t, y_{t-1}, X, w) = Math.exp(w * phi(X, y_t, y_{t-1}))
 */ 
CRF.prototype.calcPsi = function(node, leftNode)
{
	var prod = 0; // product
  var feature;
	
	// w * phi(X, y_t, y_t-1)
	for (feature in this.getFeature(node, leftNode)) {
		prod += this.weights[feature];
	}
	
	return Math.exp(prod);
}



/////////////////////// NODE FUNCTIONS ////////////////////////

/**
 * make a new node
 */
function CRFNode(x, y, leftNode) {
	var self = this instanceof CRFNode
           ? this
           : Object.create(CRFNode.prototype);
	
	self.x = x; // observed
	self.y = y; // label
	self.left = leftNode;
  self.right = null;
	self.alpha = 0;
	self.beta = 0;
	self.maxLeftNode = null;
	
	self.next = null; // pointer to a node whose x is same and y is not same
	if(typeof leftNode === "object" && leftNode !== null){
		leftNode.right = self;
	}

  return self;
}

CRFNode.prototype.isBOSNode = function(){
  if(this.x === "__BOS__" && this.y === "__BOSLABEL__" ){
    return true;
  }else{
    return false;
  }
}

CRFNode.prototype.isEOSNode = function(){
  if(this.x === "__EOS__" && this.y === "__EOSLABEL__"){
    return true;
  }else{
    return false;
  }
}

CRF.prototype.createBOSNode = function(){
  return new CRFNode("__BOS__", "__BOSLABEL__", null);
}

CRF.prototype.createEOSNode = function(left){
  return new CRFNode("__EOS__", "__EOSLABEL__", left);
}


/**
 * calc this node's alpha value
 */
CRFNode.prototype.calcAlpha = function(crf) {
	var leftNode;
	this.alpha = 0;
	
	if(this.isBOSNode()){
		this.alpha = 1;
	}else{
		// sum all psi(label,leftLabel)*leftAlpha(leftLabel)
		for(leftNode = this.left; leftNode != null; leftNode = leftNode.next){
			if(leftNode.alpha == 0){
				leftNode.calcAlpha(crf);
			}
			
			var psi = crf.calcPsi(this, leftNode);
			this.alpha += psi*leftNode.alpha;
		}
	}
}

/**
 * calc this node's beta value
 */
CRFNode.prototype.calcBeta = function(crf) {
	var rightNode;
	this.beta = 0;

	if(this.isEOSNode()){
		this.beta = 1;
	}else{
		// sum all psi(label,rightLabel)*rightBeta(rightLabel)
		for(rightNode = this.right; rightNode !== null; rightNode = rightNode.next){
			if(rightNode.beta === 0){
				rightNode.calcBeta(crf);
			}
			var psi = crf.calcPsi(rightNode, this);
			this.beta += psi*rightNode.beta;
		}
	}
}

///////////////////////// presentation //////////////////////////////

/**
 * test: show all features as a table
 */
CRF.prototype.genFeatureTable = function() {
	var html = "<table border='2'>";
	var feature;
	
	html += "<tr><th>feature</th></tr>";
	for (feature = 0; feature < this.featureNum; feature++) {
		html += "<tr><td>" + this.id2feature[feature] + "</td></tr>";
	}
	html += "</table>";

	return html;
}

/**
 * show all feature-weight map as a table
 */
CRF.prototype.genWeightsTable = function() {
	var html = '<table class="weight-table" border="2">';
	var feature;
	
	html += '<tr><th>feature</th><th>value</th></tr>';
	for (feature = 0; feature < this.featureNum; feature++){
		html += "<tr><td>" + this.id2feature[feature] + "</td><td>" + this.weights[feature] + "</td></tr>";
	}
	html += "</table>";
	
	return html;
}

/**
 * show viterbi algorithm result as a table
 */
CRF.prototype.genViterbiResultTable = function() {
	var html = '<table class="viterbi-table" border="2">';
	var node;
	var baseNode;

	// convert lattice to 2D array
	var colsArray = [];
	var colIdx = 0, nodeIdx = 0;

	for (baseNode = this.lastPredResult.bosNode; baseNode != null; baseNode = baseNode.right, colIdx++) {
		colsArray[colIdx] = [];
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
	for (nodeIdx = this.labelNum-1; 0 <= nodeIdx; nodeIdx--) {
		html += "<tr>";
		if(nodeIdx == this.labelNum-1){
			node = colsArray[0][0];
			html += this.genNodeInfoTD(node, true);
		}
		for (colIdx = 1; colIdx < colsArray.length-1; colIdx++) {
			node = colsArray[colIdx][nodeIdx];
			html += this.genNodeInfoTD(node);
		}
		if(nodeIdx == this.labelNum-1){
			node = colsArray[colsArray.length-1][0];
			html += this.genNodeInfoTD(node, true);
		}
		html += "</tr>";
	}
	
	html += "</table>";
	
	return html;
}

CRF.prototype.genNodeInfoTD = function(node, doRowSpan) {
	var td;

	if(doRowSpan){
		td = '<td rowspan="' + this.labelNum + '">';
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



/**
 * jscrf.js
 *
 * a Conditional Random Fields program by JavaScript
 *
 * @version 0.1.0
 * @author Mas_Hama (Masafumi Hamamoto)
 *
 */
"use strict";

// for Node.js application
if (typeof exports !== "undefined") {
  exports.createCRF = function(option) {
    return new CRF(option);
  }
}

/////////////////////// constructor ///////////////////////////

function CRF(option)
{
  var self = this instanceof CRF
           ? this
           : Object.create(CRF.prototype);
  option = option || {};

  self.weights = [];
  self.feature2id = {};
  self.id2feature = [];
  self.learningDataSet = [];
  self.labelList = {};
  self.lastPredResult = {}; // for genViterbiResultTable
  self.labelNum = 0;
  self.featureNum = 0;

  self.setOption(option);

  return self;
}

CRF.prototype.createBOSNode = function(){
  return new CRFNode("__BOS__", "__BOSLABEL__", null);
}

CRF.prototype.createEOSNode = function(left){
  return new CRFNode("__EOS__", "__EOSLABEL__", left);
}

//
// set learning options 
//
CRF.prototype.setOption = function(option){

  // learning weight
  if(option.epsilon){
    if(typeof option.epsilon !== "number" || option.epsilon <= 0){
      throw "invalid option.epsilon";
    }else{
      this.epsilon = option.epsilon;
    }
  }else if (this.epsilon === undefined){
    this.epsilon = 0.01;
  }

  // limit of iteration
  if(option.maxIteration){
    if(typeof option.maxIteration !== "number" || option.maxIteration <= 0){
      throw "invalid option.maxIteration";
    }else{
      this.maxIteration = option.maxIteration;
    }
  } else if (this.maxIteration === undefined) {
	 this.maxIteration = 10;
  }

  // difference threshold in weight learning process
  if (option.threshold) {
    if(typeof option.threshold !== "number" || option.threshold <= 0){
      throw "invalid option.threshold";
    }else{
      this.threshold = option.threshold;
    }
  } else if (this.threshold === undefined) {
	  this.threshold = 0.0001;
  }

  // using feature
  // TODO: feature template
  if (option.usefeature) {
    if(typeof option.usefeature !== "object") {
      throw "invalid option.usefeature";
    }else{
      this.usefeature = {};
      if (Object.keys(option.usefeature).length === 0) {
        throw "option.usefeature is empty";
      } else {
        for (var key in option.usefeature) {
          this.usefeature[key] = option.usefeature[key];
        }
      }
    }
  } else if (this.usefeature === undefined) {
	  this.usefeature = {
      "U00": true,
      "U01": true,
      "U02": true,
      "U03": true,
      "U04": true,
      "U05": true,
      "U06": true,
      "U07": true,
      "U08": true,
      "U09": true,
      "T": false
    };
  }


  // option 'verbose' is for Node.js application !
  if (option.verbose) {
    this.verbose = option.verbose;
  } else if (this.verbose === undefined) {
    this.verbose = false;
  }
}


///////////////////////// prediction /////////////////////////////

/**
 * predict input sequence represented by string
 * input string must be separated by \n and each line is assumed as an observation
 */
CRF.prototype.predictLabels = function(input) {
  var lines = input.split(/\n/);
	var line;
	var i;
	var data;
	var seqStr = "";
	var x = []; // observations of a sentence
  var allX = [];
	
	for(i = 0; i < lines.length; i++) {
		line = lines[i];
		line = line.replace(/^\s+/, "");
    // 2014/02/16: 文区切りは空行に変更
		if (line === "") {
      if (0 < x.length) {
        allX.push(x);
        x = [];
		  }
    } else {
		  x.push(line);
    }
	}

	var estimatedSeq = [];
  for (i = 0; i < allX.length; i++) {
    estimatedSeq.push(this.predict(allX[i]));
	  this.lastPredResult.xlength = x.length;
	  this.lastPredResult.estimated = estimatedSeq;
  }

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
	var lastNode, node, newNode, leftNode;
	var bosNode = this.createBOSNode();
	var leftNode = bosNode;
  var label;
  var eosNode;
  var firstNode;
  var baseNode;
	
  // for each observation, make nodes with each label
	for (i = 0; i < x.length; i++) {
		lastNode = null;
		for (label in this.labelList) {
			newNode = new CRFNode(x[i], label, leftNode); // a candidate node
			this.makeFeature(newNode, false); // without feature table updating

      if (lastNode !== null) { 
        lastNode.next = newNode;
      }
//  		newNode.next = lastNode;
			lastNode = newNode;
		}
		leftNode = leftNode.right;
	}
	eosNode = this.createEOSNode(leftNode);

  for (node = eosNode.left; node !== null; node = node.next) {
    if (node.right === null) {
      node.right = eosNode;
    }
  }

	return {bosNode: bosNode, eosNode: eosNode};
}


/**
 * do viterbi algorithm: connect from each node to the best left node
 */
CRF.prototype.viterbi = function(bosNode) {
	var baseNode, node, leftNode;
	var innerProd, score, maxScore;
	var transFeature, obsFeature;
  var i;
	var feature;
	
	bosNode.score = 0;
	for (baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right) { // move position
		for (node = baseNode; node != null; node = node.next) { // check each label at baseNode position
			maxScore = -100000000;
			
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // check all left nodes' score
				
				innerProd = 0;
//				for (feature in this.getFeature(node, leftNode)) {
//				for (feature in node.featureIdList) {
	      for (i = 0; i < node.featureIdList.length; i++) {
          feature = node.featureIdList[i];
          if (this.weights[feature] !== undefined) {
					  innerProd += this.weights[feature];
          }
          // 素性がない場合(学習データに存在しないobservation)は無視
          if (this.verbose) console.log("in viterbi ... feature: " + this.id2feature[feature] + ", weights: " + this.weights[feature]);
				}
		
        if (this.verbose) console.log("in viterbi ... here = [x: " + node.x + ", y: " + node.y + "], left: [x: " + leftNode.x + ", y: " + leftNode.y + "], innerProd: " + innerProd + ", leftNode.score " + leftNode.score);
				score = innerProd + leftNode.score;
        if (this.verbose) console.log("score: " + score);

				if(maxScore < score){
//          console.log("score updated: " + score);
					maxScore = score;
					node.score = score;
					node.maxLeftNode = leftNode;
				}
			}
		}
	}
}


///////////////////////// TRAINING ////////////////////////////

///////////////// DATA READING /////////////////

/**
 * parse a string of training data
 */
CRF.prototype.setTrainingData = function(trainingStr) {
	var lines = trainingStr.split(/\n/);
	var line;
	var newNode;
	var lastNode;
	var x, y;
	var tuple;
	var lineCount;
	var bosNode;
	var eosNode;

  // reset feature infomation
  this.feature2id = {};
  this.id2feature = [];
  this.learningDataSet = [];
  this.labelList = {};
  this.lastPredResult = {}; // for genViterbiResultTable
  this.labelNum = 0;
  this.featureNum = 0;

	var isBOS = true;
	for(lineCount = 0; lineCount < lines.length; lineCount++){
		line = lines[lineCount].replace(/^\s+/, "").replace(/\s+$/, "");
			
		if (line === "") { // 14/02/19: 文区切りを空白に
      if (isBOS) continue;

    	eosNode = this.createEOSNode(lastNode);
  	  this.learningDataSet.push({
				bosNode: bosNode,
				eosNode: eosNode
			});
			
			this.makeFeature(eosNode);

			isBOS = true;
			
		} else {
			if (isBOS) { // init bosNode
				bosNode = this.createBOSNode();
				lastNode = bosNode;
				isBOS = false;
			}
			
			tuple = line.split(/\s+/);
			y = tuple.pop();
			x = tuple;
			newNode = new CRFNode(x, y, lastNode);
			lastNode = newNode;
		
			if (!this.labelList[y]) {
				this.labelList[y] = 1;
				this.labelNum++;
			}
			
			this.makeFeature(newNode);
		}
	}

  // add not appeared labels to each observations
  this.addDummyNode();

}


CRF.prototype.addDummyNode = function() {
  var label;
  var dataIdx;
  var node, realNode, dummyNode, bosNode, eosNode;

  for (dataIdx = 0; dataIdx < this.learningDataSet.length; dataIdx++){
	  bosNode = this.learningDataSet[dataIdx].bosNode;
	  eosNode = this.learningDataSet[dataIdx].eosNode;

    for (realNode = bosNode.right; !realNode.isEOSNode(); realNode = realNode.right) {
      node = realNode;
		  for (label in this.labelList) {
        if (label === realNode.y) continue;
        node.next = new CRFNode(realNode.x, label, realNode.left, realNode.right);
        node.next.setAsDummy();
        this.makeFeature(node.next)
        node = node.next;
      }
    }
    
    // set .right property of left nodes of eosNode
    for (node = eosNode.left; node !== null; node = node.next) {
      if (node.right === null) {
        node.right = eosNode;
      }
    }

    // add transition feature between eos and lefts of eos
    this.addEOSFeature(eosNode);
  }
}

CRF.prototype.resetTraining = function() {
  var node, bosNode, baseNode;
  var dataIdx;

  for (dataIdx = 0; dataIdx < this.learningDataSet.length; dataIdx++){
	  bosNode = this.learningDataSet[dataIdx].bosNode;

    for (baseNode = bosNode; baseNode != null; baseNode = baseNode.right) {
      for (node = baseNode; node != null; node = node.next) {
        node.alpha = 0;
	      node.beta = 0;
        node.maxLeftNode = null;
      }
    }
  }
}

///////////////// FEATURE /////////////////

/**
 * initialize weight vector
 */
CRF.prototype.initWeights = function() {
	var feature;
	this.weights = [];
	for (feature = 0; feature < this.featureNum; feature++) {
//		this.weights[feature] = 0;
		this.weights[feature] = 1/Math.sqrt(this.featureNum);
	}
}

/**
 * make features of given node and set feature ID list to the node
 *
 * The argument "update" controls whether the feature table in the crf
 * object should be updated (training mode) or not (predict mode)
 */
CRF.prototype.makeFeature = function(node, update) {
  // currently, features are just only two kinds: <x, y> and <y, y-1>
  var leftx, leftleftx;
  var x = node.x[0];
  var rightx, rightrightx;

  if (update === undefined) update = true;

  // transition feature
  if (this.usefeature["T"]) {
    this.appendAFeature(node, "T:" + node.left.y + "#" + node.y, update);
  }
	
  // current observation only
  if (this.usefeature["U00"]) {
    this.appendAFeature(node, "U00:" + x + "#" + node.y, update);
  }

  // left observation only
  if (node.left !== null) {
    leftx = node.left.x[0];
    if (this.usefeature["U01"]) {
      this.appendAFeature(node, "U01:" + leftx + "#" + node.y, update);
    }
    if (this.usefeature["U02"]) {
      this.appendAFeature(node, "U02:" + leftx + "/" + x + "#" + node.y, update);
    }

    if (node.left.left !== null) {
      leftleftx = node.left.left.x[0];
      if (this.usefeature["U03"]) {
        this.appendAFeature(node, "U03:" + leftleftx + "#" + node.y, update);
      }
      if (this.usefeature["U04"]) {
        this.appendAFeature(node, "U04:" + leftleftx + "/" + leftx + "/" + x + "#" + node.y, update);
      }
    }
  }

  // right observation
  if (node.right !== null) {
    rightx = node.right.x[0];
    if (this.usefeature["U05"]) {
      this.appendAFeature(node, "U05:" + rightx + "#" + node.y, update);
    }
    if (this.usefeature["U06"]) {
      this.appendAFeature(node, "U06:" + x + "/" + rightx + "#" + node.y, update);
    }

    if (node.right.right !== null) {
      rightrightx = node.right.right.x[0];
      if (this.usefeature["U07"]) {
        this.appendAFeature(node, "U07:" + rightrightx + "#" + node.y, update);
      }
      if (this.usefeature["U08"]) {
        this.appendAFeature(node, "U08:" + x + "/" + rightx + "/" + rightrightx + "#" + node.y, update);
      }
    }
  }

  if (node.left !== null && node.right !== null) {
    if (this.usefeature["U09"]) {
      this.appendAFeature(node, "U09:" + leftx + "/" + x + "/" + rightx + "#" + node.y, update);
    }
  }
}

CRF.prototype.appendAFeature = function(node, feature, update) {
  if((!this.feature2id[feature]) && update){
    this.feature2id[feature] = this.featureNum;
    this.id2feature.push(feature);
	  this.featureNum++;
  }
  node.featureIdList.push(this.feature2id[feature]);
}

/**
 * add transition features between EOS and lefts of EOS
 */
CRF.prototype.addEOSFeature = function(eosNode) {
	var leftNode;
	for(leftNode = eosNode.left; leftNode != null; leftNode = leftNode.next){
    if (this.usefeature["T"]) {
      // always update the feature table
      this.appendAFeature(eosNode, "T:" + leftNode.y + "#" + eosNode.y, true);
    }
	}
}


///////////////// LEARNING MAIN /////////////////

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
	var oldWeights;
  var prevScore = 0; // score of last iteration
  var score = 0; // score of current iteration
  var weightL2Norm = 0;

  var baseNode;

  this.resetTraining();
	this.initWeights();

	var iteCount;
	for (iteCount = 1; iteCount <= this.maxIteration; iteCount++) {
		oldWeights = []
    prevScore = score;
    score = 0;
    if (this.verbose) console.log("### iteration " + iteCount + " ###");
		
		for (feature = 0; feature < this.featureNum; feature++) {
			deltaL[feature] = 0;
			oldWeights[feature] = this.weights[feature];
		}
		
    // read each sequence(answer) and check
		for (dataIdx = 0; dataIdx < this.learningDataSet.length; dataIdx++){
      if (this.verbose) console.log("## data No. " + dataIdx + " ##");
			bosNode = this.learningDataSet[dataIdx].bosNode;
			eosNode = this.learningDataSet[dataIdx].eosNode;

      if (this.verbose) {
        for(baseNode = bosNode; baseNode != null; baseNode = baseNode.right){ 
          console.log("baseNode: [x: " + baseNode.x + ", y: " + baseNode.y + "]");
          for(node = baseNode.next; node != null;  node = node.next) {
            console.log("node: [x: " + node.x + ", y: " + node.y + "]");
          }
        }
      }
	
			// calc all alpha and beta
      if (this.verbose) console.log("calcAlpha");
			eosNode.calcAlpha(this);

      if (this.verbose) console.log("calcBeta");
			bosNode.calcBeta(this);
	
			// calc Z
      if (this.verbose) console.log("calcZ");
			var Z = 0;
      // 14/02/22: debug: eosNode.left to eosNode
			for (node = eosNode; node != null; node = node.next) {
				Z += node.alpha;
			}
      if (this.verbose) console.log("Z: " + Z);

      if (this.verbose) console.log("calc allProbPhi");
			// calc Sum_{Y}(P(Y|X)*phi(X, Y))
			var allProbPhi = this.calcAllProbPhi(bosNode, Z);

			// calc phi(X, Y) ... merely get each feature's occurence
			// NOTE: Y is a label sequence in the training data
			var allPhi = this.calcAllPhi(bosNode);

			// add weight differentials (delta_{w}L(w^{old}))
      if (this.verbose) console.log("calc deltaL");
			for (feature = 0; feature < this.featureNum; feature++) {
				deltaL[feature] += allPhi[feature] - allProbPhi[feature];
        if (this.verbose) console.log("feature: " + feature + " (\"" + this.id2feature[feature] + "\"), current weight: " + this.weights[feature] + ", allPhi: " + allPhi[feature] + ", allProbPhi: " + allProbPhi[feature] + ", deltaL: " + deltaL[feature]);
      }

      // calc score of current weight
      score += this.calcScore(bosNode, Z);
		} // end of all data
		
		// update new weights
    if (this.verbose) console.log("update");
    weightL2Norm = 0;
		for (feature = 0; feature < this.featureNum; feature++) {
			this.weights[feature] = oldWeights[feature] + this.epsilon*deltaL[feature];
      weightL2Norm += this.weights[feature]*this.weights[feature];
		}

    // Normalize weights
    if (this.verbose) console.log("Normalize");
	  for (feature = 0; feature < this.featureNum; feature++) {
   	  this.weights[feature] /= Math.sqrt(weightL2Norm);
      if (this.verbose) console.log("feature: " + this.id2feature[feature] + ", new weight: " + this.weights[feature] + ", deltaL: " + deltaL[feature] + "\n");
	  }

    if (this.verbose) console.log("score: " + score + ", prevScore: " + prevScore + ", abs: " + Math.abs(score - prevScore)  + ", thres: " + this.threshold);

		// check converged
		if (Math.abs(score - prevScore) < this.threshold) {
      if (this.verbose) console.log("learning finished");
      break;
    }
	}
}

CRF.prototype.calcScore = function(bosNode, Z) {
  var score = 0;
  var node, leftNode;
  var logprob = 0;
  var logprob_tmp = 0;
  
  for (node = bosNode.right; node !== null; node = node.right) {
    leftNode = node.left;
    logprob += Math.log(this.calcPsi(this, node, leftNode));
  }

  score = logprob - Math.log(Z);
  if (10000 < Math.abs(score))  {
    if (this.verbose) console.log("score: " + score + "\n");
    for (node = bosNode; node !== null; node = node.right) {
      if (node !== bosNode) {
        if (this.verbose) console.log(node.x.join(""));
        logprob_tmp += Math.log(this.calcPsi(this, node, node.left));
        var tmp = "(psi: " + this.calcPsi(this, node, node.left) + ")<logprob: " + logprob_tmp + ">";
        if (this.verbose) console.log(tmp);
      }
    }
    if (this.verbose) console.log("logprob: " + logprob + ", Z: " + Z + ", logZ: " + Math.log(Z) + "\n");

    process.exit(1);
  }

  return score;
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
  var i;
	var feature;
	var allProbPhi = [];

	for (feature = 0; feature < this.featureNum; feature++){
		allProbPhi[feature] = 0;
	}

	// calc conditional merginal probability  P(y_t, y_{t-1}|X)
	for(baseNode = bosNode.right; baseNode != null; baseNode = baseNode.right){ // move t
		for (node = baseNode; node != null; node = node.next) { // change t's label
			for (leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // all left (change (t-1)'s label)

				psi = this.calcPsi(this, node, leftNode);
				prob = (psi * leftNode.alpha * node.beta) / Z;

        if (this.verbose) console.log("calc node [x: " + node.x + ", y: "+ node.y + "], left: [x: " + leftNode.x + ", y: " + leftNode.y + "], left.alpha: " + leftNode.alpha +", node.beta: " + node.beta + ", psi: " + psi + ", prob: " + prob);
				
				// add prob*phi to allProb corresponding to each feature
        for (i = 0; i < node.featureIdList.length; i++) {
          feature = node.featureIdList[i];
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
  var i;
	var transFeature, observFeature;
	
	for (feature = 0; feature < this.featureNum; feature++) {
		allPhi[feature] = 0;
	}

  // get all answer (real) features
  // NOTE: must not look dummy nodes
	for (node = bosNode.right; node != null; node = node.right) {
    for (i = 0; i < node.featureIdList.length; i++) {
      feature = node.featureIdList[i];
			allPhi[feature] += 1;
		}
	}
	
	return allPhi;
} 

/**
 * psi(y_t, y_{t-1}, X, w) = Math.exp(w * phi(X, y_t, y_{t-1}))
 */ 
CRF.prototype.calcPsi = function(crf, node, leftNode)
{
	var prod = 0; // product of weights and features(phi)
  var i;
  var feature;

	// w * phi(X, y_t, y_t-1)
	for (i = 0; i < node.featureIdList.length; i++) {
    feature = node.featureIdList[i];
		prod += crf.weights[feature];
	}

  // そのまま返すとalpha, betaの初回計算時に値が爆発してしまうので
  // 定数を引いてみる
	return Math.exp(prod-1);
//	return Math.exp(prod);
}



/////////////////////// NODE FUNCTIONS ////////////////////////

/**
 * make a new node
 */
function CRFNode(x, y, leftNode, rightNode) {
	var self = this instanceof CRFNode
           ? this
           : Object.create(CRFNode.prototype);
  var node;
	
  // observed: treated as a Array
  if (x instanceof Array) {
	  self.x = x;
  } else {
	  self.x = [x];
  }

  if (rightNode === undefined) rightNode = null;

	self.y = y; // label
	self.left = leftNode;
  self.right = rightNode;
	self.alpha = 0;
	self.beta = 0;
  self.featureIdList = [];
	self.maxLeftNode = null;
  self.__isDummy = false;

	
	self.next = null; // pointer to a node whose x is same and y is not same
	if(typeof leftNode === "object" && leftNode !== null){
    // 14/01/29: 'right' attribute set to all nodes in the left chain
    for (node = leftNode; node !== null; node = node.next) {
      if (node.right == null) node.right = self;
    }
	}

  return self;
}

CRFNode.prototype.isBOSNode = function(){
  if(this.x[0] === "__BOS__" && this.y === "__BOSLABEL__" ){
    return true;
  }else{
    return false;
  }
}

CRFNode.prototype.isEOSNode = function(){
  if(this.x[0] === "__EOS__" && this.y === "__EOSLABEL__"){
    return true;
  }else{
    return false;
  }
}

CRFNode.prototype.isDummyNode = function(){
  if(this.__isDummy){
    return true;
  }else{
    return false;
  }
}


CRFNode.prototype.setAsDummy = function() {
  this.__isDummy = true;
}

/**
 * calc this node's alpha value
 */
CRFNode.prototype.calcAlpha = function(crf, psiFunc) {
	var leftNode;
	this.alpha = 0;

  if (psiFunc === undefined) {
    psiFunc = crf.calcPsi;
  }
	
	if(this.isBOSNode()){
		this.alpha = 1;
	}else{
		// sum all psi(label,leftLabel)*leftAlpha(leftLabel)
		for(leftNode = this.left; leftNode != null; leftNode = leftNode.next){
			if(leftNode.alpha == 0){
				leftNode.calcAlpha(crf, psiFunc);
			}
			
			var psi = psiFunc(crf, this, leftNode);
			this.alpha += psi*leftNode.alpha;
		}
	}
}

/**
 * calc this node's beta value
 */
CRFNode.prototype.calcBeta = function(crf, psiFunc) {
	var rightNode;
	this.beta = 0;

  if (psiFunc === undefined) {
    psiFunc = crf.calcPsi;
  }
	
	if(this.isEOSNode()){
		this.beta = 1;
	}else{
		// sum all psi(label,rightLabel)*rightBeta(rightLabel)
		for(rightNode = this.right; rightNode !== null; rightNode = rightNode.next){
			if(rightNode.beta === 0){
				rightNode.calcBeta(crf, psiFunc);
			}

			var psi = psiFunc(crf, rightNode, this);
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
		html += "<th>" + colsArray[colIdx][0].x[0] + "</th>";
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


////////////////////////// TEST /////////////////////////////////

CRF.prototype.__test = function(verbose) {
  var x = [["x1"], ["x2"], ["x3"], ["x4"]];
  this.labelList["c1"] = 1; this.labelList["c2"] = 1;

	var latticeInfo = this.makeLattice(x);
	var bosNode = latticeInfo.bosNode;
	var eosNode = latticeInfo.eosNode;

//	eosNode.__calcAlpha(this);
	eosNode.calcAlpha(this, this.__calcPsi);
	bosNode.calcBeta(this, this.__calcPsi);

  if (this.debug) {
	  for (var baseNode = bosNode; baseNode !== null; baseNode = baseNode.right) {
  		for (var node = baseNode; node !== null; node = node.next) {
  			  console.log("x: " + node.x[0] + ", y: " + node.y + ", alpha: " + node.alpha + ", beta: " + node.beta);
   		}
  	}
  }

  var Z = 0;
	for (node = eosNode.left; node != null; node = node.next) {
    Z += node.alpha;
	}
  if (verbose) console.log("Z: " + Z);
  if (Z !== 0.084) {
    console.log("!!! Z check failed! answer: 0.084, Z: " + Z);
    return false;
  }

	for (var baseNode = bosNode; baseNode != null; baseNode = baseNode.right) {
		for (var node = baseNode; node != null; node = node.next) {
			if (verbose) console.log("x: " + node.x[0] + ", y: " + node.y + ", alpha: " + node.alpha + ", beta: " + node.beta);
			for (var leftNode = node.left; leftNode != null; leftNode = leftNode.next) { // all left (change (t-1)'s label)
			  var psi = this.__calcPsi(this, node, leftNode);
			  var prob = psi * leftNode.alpha * node.beta / Z;
			  if (verbose) console.log("x: " + node.x[0] + ", y: " + node.y + " left y: " + leftNode.y + ", prob: " + prob);

        if (!this.__check(node, leftNode, prob)) {
			    console.log("!!! check failed!!!");
          return false;
        }
      }
    }
  }

  return true;
}

CRF.prototype.__check = function(node, leftNode, prob)
{
  var alphaAnswer;
  var betaAnswer;
  var probAnswer;

  if (node.x[0] === "__BOS__" && node.y == "__BOSLABEL__") {
    alphaAnswer = 1;
    betaAnswer = 0.084;
  } else if (node.x[0] === "x1" && node.y == "c1") {
    alphaAnswer = 1;
    betaAnswer = 0.036;
  } else if (node.x[0] === "x1" && node.y == "c2") {
    alphaAnswer = 1;
    betaAnswer = 0.048;
  } else if (node.x[0] === "x2" && node.y == "c1") {
    alphaAnswer = 0.5;
    betaAnswer = 0.12;
  } else if (node.x[0] === "x2" && node.y == "c2") {
    alphaAnswer = 0.2;
    betaAnswer = 0.12;
  } else if (node.x[0] === "x3" && node.y == "c1") {
    alphaAnswer = 0.14;
    betaAnswer = 0.5;
    if (leftNode.y == "c1") {
      probAnswer = 0.595;
    } else if (leftNode.y == "c2") {
      probAnswer = 0.238;
    }
  } else if (node.x[0] === "x3" && node.y == "c2") {
    alphaAnswer = 0.07;
    betaAnswer = 0.2;
    if (leftNode.y == "c1") {
      probAnswer = 0.119;
    } else if (leftNode.y == "c2") {
      probAnswer = 0.048;
    }
  } else if (node.x[0] === "x4" && node.y == "c1") {
    alphaAnswer = 0.049;
    betaAnswer = 1;
  } else if (node.x[0] === "x4" && node.y == "c2") {
    alphaAnswer = 0.035;
    betaAnswer = 1;
  } else if (node.x[0] === "__EOS__" && node.y == "__EOSLABEL__") {
    alphaAnswer = 0.084;
    betaAnswer = 1;
  }
  
  if (Math.round(node.alpha*1000)/1000 !== alphaAnswer) {
    console.log("alpha test failed");
    return false;
  }
  if (Math.round(node.beta*1000)/1000 !== betaAnswer) {
    console.log("beta test failed");
    return false;
  }

  if (probAnswer && Math.round(prob*1000)/1000 !== probAnswer) {
    console.log("prob test failed");
    return false;
  }

  return true;
}


CRF.prototype.__calcPsi = function(crf, node, leftNode)
{
  if (leftNode.isBOSNode()) { // node.x == "x1"
    return 1.0;
  } else if (node.x[0] === "x2") {
    if (node.y == "c1") {
      if (leftNode.y == "c1") {
        return 0.2;
      } else if (leftNode.y == "c2") {
        return 0.3;
      }
    } else if (node.y == "c2") {
      return 0.1;
    }
  } else if (node.x[0] === "x3") {
    if (node.y == "c1") {
      return 0.2;
    } else if (node.y == "c2") {
      return 0.1;
    }
  } else if (node.x[0] === "x4") {
    if (node.y == "c1") {
      if (leftNode.y == "c1") {
        return 0.3;
      } else if (leftNode.y == "c2") {
        return 0.1;
      }
    } else if (node.y == "c2") {
      if (leftNode.y == "c1") {
        return 0.2;
      } else if (leftNode.y == "c2") {
        return 0.1;
      }
    }
  } else if (node.isEOSNode()) {
    return 1.0;
  }
}


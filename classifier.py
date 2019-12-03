import time 

def loadTrainingData():
	#load face training images into array 
	f = open("data/facedata/facedatatrain")
	lines = f.read().splitlines() 
	faceTrainingImages = []
	i = 0
	tempArray = []
	for line in lines:
		tempArray.append(line)
		i += 1
		if(i == 70):
			faceTrainingImages.append(tempArray)
			i = 0
			tempArray = []

	#load face training labels into array 
	f = open("data/facedata/facedatatrainlabels")
	lines = f.read().splitlines()
	faceTrainingLabels = []
	for line in lines:
		faceTrainingLabels.append(line)

	#load digit training images into array 
	f = open("data/digitdata/trainingimages")
	lines = f.read().splitlines() 
	digitTrainingImages = []
	i = 0
	tempArray = []
	for line in lines:
		tempArray.append(line)
		i += 1
		if(i == 28):
			digitTrainingImages.append(tempArray)
			i = 0
			tempArray = []

	#load digit training labels into array 
	f = open("data/digitdata/traininglabels")
	lines = f.read().splitlines()
	digitTrainingLabels = []
	for line in lines:
		digitTrainingLabels.append(line)

	return faceTrainingImages, faceTrainingLabels, digitTrainingImages, digitTrainingLabels

def loadTestingData():
	#load face training images into array 
	f = open("data/facedata/facedatatest")
	lines = f.read().splitlines() 
	faceTestingImages = []
	i = 0
	tempArray = []
	for line in lines:
		tempArray.append(line)
		i += 1
		if(i == 70):
			faceTestingImages.append(tempArray)
			i = 0
			tempArray = []

	#load face training labels into array 
	f = open("data/facedata/facedatatestlabels")
	lines = f.read().splitlines()
	faceTestingLabels = []
	for line in lines:
		faceTestingLabels.append(line)

	#load digit training images into array 
	f = open("data/digitdata/testimages")
	lines = f.read().splitlines() 
	digitTestingImages = []
	i = 0
	tempArray = []
	for line in lines:
		tempArray.append(line)
		i += 1
		if(i == 28):
			digitTestingImages.append(tempArray)
			i = 0
			tempArray = []

	#load digit training labels into array 
	f = open("data/digitdata/testlabels")
	lines = f.read().splitlines()
	digitTestingLabels = []
	for line in lines:
		digitTestingLabels.append(line)

	return faceTestingImages, faceTestingLabels, digitTestingImages, digitTestingLabels

def trainFacePerceptron(images, labels, trainingSize):
	start = time.time()
	weights = [0] * ((len(images[0]) * len(images[0][0])))
	bias = 0
	end = int((float(trainingSize/100.0))*len(images))
	wchange = True
	while wchange:
		wchange = False 
		for image in images[0:end]:
			#Function to return f(x) for each image
			val = trainOnImage(image, weights, bias)
			if ((val >= 0) and (labels[images.index(image)] == '0')):
				weights,bias = updateWeights(image, weights, bias, -1)
				wchange=True
			elif (val < 0 and (labels[images.index(image)]) == '1'): 
				weights,bias = updateWeights(image, weights, bias, 1)
				wchange=True
	end = time.time()
	runtime = end - start
	return weights, bias, runtime 

def trainDigitPerceptron(images, labels, trainingSize):
	start = time.time()
	weights = []
	i = 0
	#Initiliaze weight sets
	while(i < 10):
		tempArray = [0] * ((len(images[0]) * len(images[0][0])))
		weights.append(tempArray)
		i += 1
	#Initialize bias set
	biasSet = [0] * 10
	end = int((float(trainingSize/100.0))*len(images))
	wchange = True
	while wchange:
		wchange = False 
		for image in images[0:end]:
			vals = [0] * 10
			#Run on all 10 digit perceptrons 
			j = 0
			while(j < 10):
				vals[j] = trainOnImage(image, weights[j], biasSet[j])
				j += 1
			if (str(vals.index(max(vals))) != labels[images.index(image)]):
				#Update weights
				updateWeights(image, weights[vals.index(max(vals))], biasSet[vals.index(max(vals))],-1)
				updateWeights(image, weights[int(labels[images.index(image)])], biasSet[int(labels[images.index(image)])], 1)
	end = time.time()
	runtime = end - start
	return weights, biasSet, runtime

def updateWeights(image, weights, bias, change):
	if(change > 0): #Increase weights
		bias += 1
		k=0
		for i in image:
			for j in i:
				if(j != ' '):
					weights[k] += 1 
				k += 1
	else: #Decrease weights
		bias -= 1
		k=0
		for i in image:
			for j in i:
				if(j != ' '):
					weights[k] -= 1 
				k += 1
	return weights,bias

def trainOnImage(image, weights, bias):
	fValue = 0
	fValue += bias
	k=0;
	for i in image:
		for j in i:
			if(j == ' '):
				fValue += 0
			else: 
				fValue += weights[k]
			k += 1
	return fValue

def testFacePerceptron(images, weights, bias, labels, trainingSize, runtime):
	correct = 0
	incorrect = 0
	for image in images:
		val = trainOnImage(image, weights, bias)
		if(val >= 0):
			if(labels[images.index(image)] == '1'):
				correct += 1
			else:
				incorrect += 1
		else: 
			if(labels[images.index(image)] == '0'):
				correct += 1
			else:
				incorrect += 1
	percentCorrect = float(correct/float(correct+incorrect))*100
	percentIncorrect = float(incorrect/float(correct+incorrect))*100
	print("Training Set Size: " + str(trainingSize) + "%")
	print("Runtime: " + str(runtime))
	print("Correct: " + str(percentCorrect) + "%")
	print("Incorrect: " + str(percentIncorrect) + "%")

def testDigitPerceptron(images, weights, biases, labels, trainingSize, runtime):
	correct = 0
	incorrect = 0
	for image in images:
		vals = [0] * 10
		#Run on all 10 digit perceptrons 
		j = 0
		while(j < 10):
			vals[j] = trainOnImage(image, weights[j], biases[j])
			j += 1
		if (str(vals.index(max(vals))) != labels[images.index(image)]):
			incorrect += 1
		else: 
			correct += 1
	percentCorrect = float(correct/float(correct+incorrect))*100
	percentIncorrect = float(incorrect/float(correct+incorrect))*100
	print("Training Set Size: " + str(trainingSize) + "%")
	print("Runtime: " + str(runtime))
	print("Correct: " + str(percentCorrect) + "%")
	print("Incorrect: " + str(percentIncorrect) + "%")

def trainFaceNaive(images, labels, trainingSize):
	start = time.time()
	faces = 0
	for label in labels:
		if label == '1':
			faces += 1
	nonfaces = len(labels) - faces
	priorFace = float(faces)/float(len(labels))
	priorNotFace = float(nonfaces)/float(len(labels))
	end = int((float(trainingSize/100.0))*len(images))
	#load face table with probabilities of features containing pixel 
	featuresFace = [0] * ((len(images[0]) * len(images[0][0])))
	for image in images[0:end]: 
		if(labels[images.index(image)] == '1'):
			k = 0
			for i in image:
				for j in i:
					if(j != ' '):
						featuresFace[k] += 1
					k += 1
	for l in featuresFace: 
		if (l != 0):
			l = float(l)/float(faces)
		else: 
			l = float(1)/float(faces)

	#load non face table 
	featuresNotFace = [0] * ((len(images[0]) * len(images[0][0])))
	for image in images[0:end]:
		if(labels[images.index(image)] == '0'):
			h = 0
			for n in image:
				for p in n:
					if (p != ' '):
						featuresNotFace[h] += 1
					h += 1
	for q in featuresNotFace:
		if (q != 0):
			q = float(q)/float(nonfaces)
		else:
			q = float(1)/float(nonfaces)

	end = time.time()
	runtime = end - start 
	return featuresFace, featuresNotFace, runtime

def testFaceNaive(images, labels, featureTableFace, featureTableNotFace, trainingSize, runtime):
	correct = 0
	incorrect = 0
	for image in images:
		pFace = evaluateImage(image, featureTableFace)
		pNotFace = evaluateImage(image, featureTableNotFace)
		if(pFace >= pNotFace):
			if(labels[images.index(image)] == '0'):
				incorrect += 1
			else:
				correct += 1
		else:
			if(labels[images.index(image)] == '1'):
				incorrect += 1
			else:
				correct += 1
	percentCorrect = float(correct/float(correct+incorrect))*100
	percentIncorrect = float(incorrect/float(correct+incorrect))*100
	print("Training Set Size: " + str(trainingSize) + "%")
	print("Runtime: " + str(runtime))
	print("Correct: " + str(percentCorrect) + "%")
	print("Incorrect: " + str(percentIncorrect) + "%")

def evaluateImage(image, featureTable):
	val = 1
	k = 0
	for j in image:
		for i in j:
			if (i != ' '):
				val *= featureTable[k]
			else: 
				val *= (1 - featureTable[k])
			k += 1
	return val


if __name__ == "__main__":

	#Load training images and labels into arrays 
	fImages, fLabels, dImages, dLabels = loadTrainingData()

	#Load testing images and labels into arrays
	fTestImages, fTestLabels, dTestImages, dTestLabels = loadTestingData()

	#Prompt user for data type (either face classificaion of digit classification)
	while True:
		dataType = raw_input("Enter F for Faces or D for Digits.\n")
		if(dataType != 'f' and dataType != 'd'):
			print("Improper input Try again.\n")
		else: 
			break 

	while True:
		classifier = raw_input("Enter P for Perceptron or N for Naive Bayes.\n")
		if(classifier != 'p' and classifier != 'n'):
			print("Improper input Try again.\n")
		else: 
			break

	while True: 
		trainingSize = input("Enter the percentage of training set images to be used (must be multiple of 10).\n")
		if((trainingSize % 10) != 0 or trainingSize > 100):
			print("Improper input Try again.\n")
		else: 
			break 

	if(dataType == 'f'):
		if(classifier == 'p'):
			weights, bias, runtime = trainFacePerceptron(fImages, fLabels, trainingSize)
			testFacePerceptron(fTestImages, weights, bias, fTestLabels, trainingSize, runtime)
		elif(classifier == 'n'):
			trainFaceNaive(fImages, fLabels, trainingSize)
			featureTableFace, featureTableNotFace, runtime = trainFaceNaive(fImages, fLabels, trainingSize)
			testFaceNaive(fTestImages, fTestLabels, featureTableFace, featureTableNotFace, trainingSize, runtime)	
	elif(dataType == 'd'):
		if(classifier == 'p'):
			weights, biases, runtime = trainDigitPerceptron(dImages, dLabels, trainingSize)
			testDigitPerceptron(dTestImages, weights, biases, dTestLabels, trainingSize, runtime)



	








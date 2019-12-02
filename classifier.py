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

def trainPerceptron(images, labels, trainingSize):
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
	return weights, bias

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

def testPerceptron(images, weights, bias, labels, trainingSize):
	correct = 0
	incorrect = 0
	for image in images:
		val = trainOnImage(image, weights, bias)
		if(val >= 0):
			if(labels[images.index(image)] == '1'):
				print("correct prediction")
				correct += 1
			else:
				print("incorrect prediction")
				incorrect += 1
		else: 
			if(labels[images.index(image)] == '0'):
				print("correct prediction")
				correct += 1
			else:
				print("incorrect prediction")
				incorrect += 1
	percentCorrect = float(correct/float(correct+incorrect))*100
	percentIncorrect = float(incorrect/float(correct+incorrect))*100

	print("Correct: " + str(percentCorrect) + "%")
	print("Incorrect: " + str(percentIncorrect) + "%")


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
			weights, bias = trainPerceptron(fImages, fLabels, trainingSize)
			testPerceptron(fTestImages, weights, bias, fTestLabels, trainingSize)
	elif(dataType == 'd'):
		if(classifier == 'p'):
			print("running Perceptron on digits")



	

























import sys
import time
start = time.strftime('%b %d %Y %H:%M:%S',time.localtime(time.time()))
import numpy
import joblib
import pybel
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
numpy.random.seed(507)

def mol_calcfp(f1):
		mols =  list(pybel.readfile("smi",f1))
		bits = [x.calcfp() for x in mols]
		titles = [x.title for x in mols]
		return bits,titles

def mol_calcfpbit(f1):
		mols =  list(pybel.readfile("smi",f1))
		bits = [x.calcfp().bits for x in mols]
		titles = [x.title for x in mols]
		return bits,titles

def cal_psore(f1,f2):
		testbit,testtitles=mol_calcfp(f1)
		cabit,catitles=mol_calcfp(f2)
		fpscore={}
		for i in range(len(testtitles)):
				atc1={}
				for j in range(len(catitles)):
						tascore=testbit[i] | cabit[j]
						atc1[catitles[j]]=tascore
				atclist= sorted(atc1.items(), key=lambda d:d[1], reverse = True)
				atc=atclist[0][1]
				matc=atclist[0][0]
				fpscore[testtitles[i]]=(matc,atc)
		return fpscore

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print ("the word: %s is not in my Vocabulary!" % word)
	return returnVec

######main######
###usage:python step5_pre_anti-SARS-CoV-2_compound.py inputfilename outputfilename
###usage:python step5_pre_anti-SARS-CoV-2_compound.py ../data/test.smi ../data/test_prediction_result.txt
inputfilename=sys.argv[1].strip()###'../data/test.smi'
outputfilename=sys.argv[2].strip() ####'../data/test_prediction_result.txt'
FPscore_dict={}
FPscoret_dict={}
SVM_dict={}
RF_dict={}
SVMa_dict={}
RFa_dict={}


f1=inputfilename
f2='../data/active_final.smi'
f3=open(outputfilename,'w')
header="Qurey\tMatch_ID\tMax_TC\tSVM_Prob\tSVM_Pre\tRF_Prob\tRF_Pre\n"
f3.write(header)

fpscore=cal_psore(f1,f2)
bits,titles=mol_calcfpbit(f1)
fingerprint=range(1,1025)
querydata=[]
for i in bits:
	returnVec=setOfWords2Vec(fingerprint,i)
	querydata.append(returnVec)
x = numpy.array(querydata)

for i in range(len(titles)):
	FPscore_dict[titles[i]]=fpscore[titles[i]][1]
	FPscoret_dict[titles[i]]=fpscore[titles[i]][0]

###SVM
svm_model=joblib.load("../data/SARS2_SVM_model_50_rbf.pkl.z")###change
y_pred= svm_model.predict(x)
y_pred1= svm_model.decision_function(x)
for i in range(len(y_pred1)):
	y_predp=y_pred1[i]
	SVM_dict[titles[i]]=y_predp
	SVMa_dict[titles[i]]=y_pred[i]

###RF
RFmodel=joblib.load("../data/SARS2_RFmodel_900.pkl.z")###change
y_pred= RFmodel.predict(x)
y_pred1= RFmodel.predict_proba(x)
for i in range(len(y_pred1)):
	y_predp=y_pred1[i][1]
	RF_dict[titles[i]]=y_predp
	RFa_dict[titles[i]]=y_pred[i]

for i in FPscore_dict.keys():
	stra='\t'.join([str(i),str(FPscoret_dict[i]),str(FPscore_dict[i]),str(SVM_dict[i]),str(SVMa_dict[i]),str(RF_dict[i]),str(RFa_dict[i])])+'\n'
	f3.write(stra)
end = time.strftime('%b %d %Y %H:%M:%S',time.localtime(time.time()))
f3.write("Start Time: " + start + "\n")
f3.write("End Time: " + end + "\n")

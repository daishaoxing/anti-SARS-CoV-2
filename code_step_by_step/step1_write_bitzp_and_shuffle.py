import pybel
import joblib
import random

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print ("the word: %s is not in my Vocabulary!" % word)
	return returnVec

def get_bits(f1):
	mols = list(pybel.readfile("smi",f1));
	titles = [x.title for x in mols]
	bits = [x.calcfp().bits for x in mols]
	return bits,titles

def numtostringlist(inlist):
	list1=[]
	for i in inlist:
		a=str(i)
		list1.append(a)
	return list1

f1=open('../data/all_data.txt','w')
f2=open('../data/all_datalabel.txt','w')
activebits,activetitles=get_bits('../data/active_final.smi')
inactivebits,inactivetitles=get_bits('../data/inactive_final.smi')

fingerprint=range(1,1025)
alldata=[]
alldatalabel=[]
for i in activebits:
	returnVec=setOfWords2Vec(fingerprint,i)
	list1=numtostringlist(returnVec)
	f1.write('\t'.join(list1)+'\n')
	f2.write('1'+'\n')
	alldata.append(returnVec)
	alldatalabel.append(1)
	
for i in inactivebits:
	returnVec=setOfWords2Vec(fingerprint,i)
	list1=numtostringlist(returnVec)
	f1.write('\t'.join(list1)+'\n')
	f2.write('0'+'\n')
	alldata.append(returnVec)
	alldatalabel.append(0)
data_and_label=[]
for i in range(len(alldata)):
	a=[alldata[i],alldatalabel[i]]
	data_and_label.append(a)
alldata=[]
alldatalabel=[]
random.shuffle(data_and_label)

for i in range(len(data_and_label)):
	alldata.append(data_and_label[i][0])
	alldatalabel.append(data_and_label[i][1])
joblib.dump(alldata,"../data/alldata.pkl.z")
joblib.dump(alldatalabel,"../data/alldatalabel.pkl.z")

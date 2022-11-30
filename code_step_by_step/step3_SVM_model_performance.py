from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy
from sklearn.externals import joblib
import threading
numpy.random.seed(507)


def run_svm_thread(x_train,x_test,y_train,y_test,k,j,cv):
		clf = svm.SVC(C=j,kernel=k,probability=False,random_state=507)
		svm_model=clf.fit(x_train, y_train)
		y_pred= svm_model.predict(x_test)
		accuracy=metrics.accuracy_score(y_test, y_pred)
		precision=metrics.precision_score(y_test, y_pred)
		recall=metrics.recall_score(y_test, y_pred)
		f1score=metrics.f1_score(y_test,y_pred)
		y_pred1=[]
		y_pred2= svm_model.decision_function(x_test)
		for i in y_pred2:
			y_pred1.append(i)
		fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred1, pos_label=1)
		roc_auc=metrics.auc(fpr, tpr)
		strb=str(j)+'\t'+str(cv)+'\t'+str(accuracy)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1score)+'\t'+str(roc_auc)+'\n'
		f1.write(strb)
		print (strb)

######main######
f1=open('../data/svm_performance_final.txt','a')
alldataMat=joblib.load('../data/alldata.pkl.z')
alldataClasses=joblib.load('../data/alldatalabel.pkl.z')

#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)
k='rbf'
C=[0.5,1]
C.extend(range(50,1001,50))
for cv in range(10):
	x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)
	for j in C:
		myargs = (x_train, x_test, y_train, y_test,k,j,cv,)
		threading.Thread(target=run_svm_thread, args=myargs).start()

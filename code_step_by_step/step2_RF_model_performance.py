from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import threading
from sklearn.ensemble import RandomForestClassifier
import numpy
numpy.random.seed(507)

def run_RF_thread(x_train,x_test,y_train,y_test,i,cv):
		clf = RandomForestClassifier(max_depth=None,n_estimators=i,max_leaf_nodes=None,min_samples_split=2,min_samples_leaf=1, random_state=507)
		RFmodel=clf.fit(x_train, y_train)
		y_pred= RFmodel.predict(x_test)
		accuracy=metrics.accuracy_score(y_test, y_pred)
		precision=metrics.precision_score(y_test, y_pred)
		recall=metrics.recall_score(y_test, y_pred)
		f1score=metrics.f1_score(y_test,y_pred)
		y_pred1=[]
		y_pred2= RFmodel.predict_proba(x_test)
		for j in y_pred2:
			y_pred1.append(j[1])
		fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred1, pos_label=1)
		roc_auc=metrics.auc(fpr, tpr)
		strb=str(i)+'\t'+str(cv)+'\t'+str(accuracy)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1score)+'\t'+str(roc_auc)+'\n'
		f1.write(strb)
		print (strb)

######main######
f1=open('../data/RF_performance_final.txt','a')
alldataMat=joblib.load('../data/alldata.pkl.z')
alldataClasses=joblib.load('../data/alldatalabel.pkl.z')

#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)
n_estimator=range(50,1001,50)

for cv in range(10):
		x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)
		for i in n_estimator:
			# run_RF_thread(x_train,x_test,y_train,y_test,i,cv)
			myargs = (x_train, x_test, y_train, y_test,i,cv,)
			threading.Thread(target=run_RF_thread, args=myargs).start()

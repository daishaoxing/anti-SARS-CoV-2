import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import joblib
numpy.random.seed(507)


#####loaddata
alldataMat=joblib.load("../data/alldata.pkl.z")
alldataClasses=joblib.load("../data/alldatalabel.pkl.z")
#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)

###########SVM
clf = svm.SVC(C=50,kernel='rbf',probability=False,random_state=507) ###change parameters C
svm_model=clf.fit(x, y)
joblib.dump(svm_model,'../data/SARS2_SVM_model_50_rbf.pkl.z')
#############RF
clf = RandomForestClassifier(max_depth=None,n_estimators=900,
                             max_leaf_nodes=None,min_samples_split=2,
                             min_samples_leaf=1, random_state=507) ###change parameters n_estimators
RFmodel=clf.fit(x, y)
joblib.dump(RFmodel,'../data/SARS2_RFmodel_900.pkl.z')

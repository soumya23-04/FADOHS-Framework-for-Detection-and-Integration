from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,sentiment_and_emotion_analysis,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_sentiment_and_emotion_analysis_Status(request):
    if request.method == "POST":

        if request.method == "POST":

            pid=request.POST.get('pid')
            post=request.POST.get('post')


        df = pd.read_csv('Datasets.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0
            elif (Label == 1):
                return 1
            elif (Label == 2):
                return 2
            elif (Label == 3):
                return 3
            elif (Label == 4):
                return 4

        df['results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Post']
        y = df['results']

        print("Post")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()
        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("SGD Classifier")
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
        sgd_clf.fit(X_train, y_train)
        sgdpredict = sgd_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, sgdpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, sgdpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, sgdpredict))
        models.append(('SGDClassifier', sgd_clf))


        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        rfpredict = rf_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, rfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, rfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, rfpredict))
        models.append(('RandomForestClassifier', rf_clf))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))


        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        post1 = [post]
        vector1 = cv.transform(post1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Hate Speech'
        elif (prediction == 1):
            val = 'Offensive Speech'
        elif (prediction == 2):
            val = 'Neutral Speech'
        elif (prediction == 3):
            val = 'Positive Speech'
        elif (prediction == 4):
            val = 'Negative Speech'

        print(val)
        print(pred1)

        sentiment_and_emotion_analysis.objects.create(
        post_id=pid,
        post=post,
        Prediction=val)

        return render(request, 'RUser/Predict_sentiment_and_emotion_analysis_Status.html',{'objs': val})
    return render(request, 'RUser/Predict_sentiment_and_emotion_analysis_Status.html')




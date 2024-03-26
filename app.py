import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import mysql.connector
from config import DB_CONFIG
df=pd.read_csv('heart.csv')
df.head()
df.shape
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier()
svc = SVC()
accuracy_score= np.mean(cross_val_score(RandomForestClassifier(),x,y,scoring='accuracy'))
n_estimators=[20,60,100,120]

# Number of features to consider at every split
max_features=[0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth=[2,4,8,None]

#Number of samples
max_samples=[0.5,0.75,1.0]
param_grid={'n_estimators':n_estimators,
           'max_features': max_features,
           'max_depth':max_depth,
           'max_samples': max_samples
           }
# print(param_grid)
rf = RandomForestClassifier()
rf_grid=GridSearchCV(estimator=rf,
                    param_grid=param_grid,
                    cv=5,
                    verbose=2,
                    n_jobs=-1)

# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

# Bootstrap samples
bootstrap = [True,False]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              'max_samples':max_samples,
              'bootstrap':bootstrap,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf
             }
# print(param_grid)
rf_grid=RandomizedSearchCV(estimator=rf,
                          param_distributions=param_grid,
                          cv=5,
                          verbose=2,
                          n_jobs=-1)


# rf_grid.fit(x_train,y_train)
rf=RandomForestClassifier(oob_score=True)
rf.fit(x_train,y_train)


rf.oob_score_

app = Flask(__name__)
app.secret_key = 'abcd21234455'  
db = mysql.connector.connect(**DB_CONFIG)
cursor = db.cursor()
# rf_grid.fit(x_train,y_train)
@app.route('/', methods=['GET', 'POST'])
def index():
  if 'loggedin' in session:   
    if request.method == 'POST':
        # Get user input from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Make predictions
        user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction =  rf.predict(user_data)
        userid=session['userid'] 
         # Insert user data and prediction into the database
        insert_user_query = """
        INSERT INTO heart_disease_prediction (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction,userid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Convert data types to built-in types
        data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach),
                int(exang), float(oldpeak), int(slope), int(ca), int(thal), int(prediction), int(userid)]
        
        cursor.execute(insert_user_query, tuple(data))
        db.commit()
        Answer =''

        if prediction == 1:
    
          if int(thal) > 4:
            Answer = "Coronary Heart Disease"
            Prescription = "Angiotensin-converting enzyme (ACE) inhibitors"
          elif int(cp) > 3:
            Answer = "Cardiac Arrest"
            Prescription = "Coronary bypass surgery"
          elif int(trestbps) > 140:
            Answer = "High Blood Pressure"
            Prescription = "Beta-blockers"
          else:
            Answer = "Arrhythmia"
            Prescription = "Procainamide (Procan, Procanbid)"
        else:
             Answer = "No heart disease detected"
             Prescription = "No prescription needed"
        return render_template('result.html', prediction=prediction,pre=Answer ,medi=Prescription)

    # Render the main page
    return render_template('index.html')
  return redirect(url_for('login'))
    # Render the main page
cv_accuracy = np.mean(cross_val_score(RandomForestClassifier(), x, y, scoring='accuracy'))


@app.route('/about')
def about():  
    if 'loggedin' in session:        
      return render_template('about.html')
    else:
     return render_template('about.html')
########################### login section ##################################

@app.route('/login', methods=['GET', 'POST'])
def login():
    type = ''
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']

        cursor.execute('SELECT * FROM users WHERE status="active" AND email = %s AND password = %s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user[0]  # Assuming user ID is at index 0 in the tuple
            session['name'] = user[1]   # Assuming user's first name is at index 1
            session['email'] = user[2]  # Assuming user's email is at index 2
            session['role'] = user[5]   # Assuming user's role is at index 3
            type= session['role']
            message = 'Logged in successfully !'
            if type == 'administrator':
             return redirect(url_for('dashboard'))
            elif type == 'User':
             return redirect(url_for('index'))
          
           
           
        else:
            message = 'Email or Password Not Match'
   
    return render_template('login.html', message=message)
########################### register section ##################################

@app.route('/register', methods=['GET', 'POST'])
def register():
    
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
     name = request.form['name']
     email = request.form['email']
     mobile = request.form['mobile']
     password = request.form['password']

    # Insert data into database
     insert_query = "INSERT INTO users (name, email, password,mobile) VALUES (%s, %s, %s, %s)"
     cursor.execute(insert_query, (name, email, password, mobile))
     db.commit()
     message = 'User Register Successfully!!'
    else:
     message = ''
    return render_template('/register.html', message=message)
########################### register section ##################################

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():
    if 'loggedin' in session:        
        return render_template("dashboard.html")
    return redirect(url_for('login'))
 ########################### Dataset  section ##################################

def generate_targetplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    colors = ["#1CA53B", "#AA1111"]
    sns.countplot(x="target", data=df, palette="mako_r")
    plt.savefig("static/target.png")
def generate_sexplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    countNoDisease = len(df[df.target == 0])
    countHaveDisease = len(df[df.target == 1])
    print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
    print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
    colors = ["#1CA53B", "#AA1111"]
    sns.countplot(x='sex', data=df, palette=colors)
    plt.xlabel("Sex (0 = female, 1= male)")
    plt.savefig("static/sexplot.png")
def generate_cpplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    sns.countplot(x="cp", data=df, palette="mako_r")
    plt.savefig("static/cpplot.png")
def generate_thalplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    sns.countplot(x="thal", data=df, palette="mako_r")
    plt.savefig("static/thalplot.png")


def generate_heartdiseaseplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('static/heartDiseaseAndAges.png')

def generate_frequencyplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig("static/Frequencyplot.png")
def generate_heartrateplot():
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.savefig("static/HeartRateplot.png")
  ########################### Algorithm section ##################################
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
output = pd.DataFrame({'Predicted':Y_pred}) # Heart-Disease yes or no? 1/0
score_logreg = score
out_logreg = output

decision_tree = DecisionTreeClassifier(max_depth=5) 
decision_tree.fit(X_train, y_train)  
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)
score_dtc = model.score(X_test, y_test)


model = KNeighborsClassifier()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
score_knc = score
out_knc = output
def generate_algorithmplot():
 plt.rcdefaults()
 
fig, ax = plt.subplots(figsize=(15, 6))
algorithms = ('Logistic Regression', 'K Neighbors Classifier', 'Random Forest Classifier', 'Decision Tree Classifier')
y_pos = np.arange(len(algorithms))
x = (score_logreg, score_knc, cv_accuracy, score_dtc) # scores
ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms)
ax.invert_yaxis() # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('Which one is the best algorithm?')
for i, v in enumerate(x):
    ax.text(v + 1, i, str(v), color='black', va='center', fontweight='normal')
plt.savefig("static/algorithmsplot.png")

  ########################### Dataset&Algorithm section ##################################
@app.route("/Dataset", methods=['GET', 'POST'])
def Dataset():
    if 'loggedin' in session:        
        generate_targetplot()
        generate_sexplot()
        generate_heartdiseaseplot()
        generate_frequencyplot()
        generate_heartrateplot()
        generate_cpplot() 
        generate_thalplot()
        return render_template("Dataset.html")
    return redirect(url_for('login'))
@app.route("/MLalgorithm", methods=['GET', 'POST'])
def MLalgorithm():
    if 'loggedin' in session:        
        generate_algorithmplot()
        
        return render_template("MLalgorithm.html",score_logreg=score_logreg,score_knc=score_knc,cv_accuracy=cv_accuracy,score_dtc=score_dtc)
    return redirect(url_for('login'))
  

########################### Prediction section ##################################

@app.route("/predictions", methods =['GET', 'POST'])
def predictions():
    if 'loggedin' in session:   
        
        cursor.execute('SELECT * FROM heart_disease_prediction  ')
        prediction = cursor.fetchall() 
         
        return render_template("prediction.html", prediction = prediction)
    return redirect(url_for('login')) 
    
@app.route("/edit_predictions", methods =['GET'])
def edit_predictions():
    if 'loggedin' in session:
        predictions_id = request.args.get('predictions_id') 
        
        cursor.execute('SELECT * FROM heart_disease_prediction WHERE id = %s', (predictions_id,))
        predictionss = cursor.fetchall() 
 
        
        return render_template("edit_predictions.html", predictionss = predictionss)
    return redirect(url_for('login'))  
    
@app.route("/save_predictions", methods =['GET', 'POST'])
def save_predictions():
    if 'loggedin' in session:    
                
        if request.method == 'POST' and 'age' in request.form and 'sex' in request.form:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])
            action = request.form['action']             
            user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            prediction =  rf.predict(user_data)
            if action == 'updatepredictions':
                predictionsid = request.form['predictionsid'] 
               
                insert_user_query = """UPDATE heart_disease_prediction SET age = %s,sex = %s,cp = %s,trestbps = %s,chol = %s,fbs = %s,restecg = %s,thalach = %s,exang = %s,oldpeak = %s,slope = %s,ca = %s,thal = %s,prediction = %s WHERE id =%s"""
        # Convert data types to built-in types
                data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach),
                int(exang), float(oldpeak), int(slope), int(ca), int(thal), int(prediction), int(predictionsid)]
                cursor.execute(insert_user_query, tuple(data))
                db.commit()
           
            return redirect(url_for('predictions'))        
        elif request.method == 'POST':
            msg = 'Please fill out the form field !'        
        return redirect(url_for('predictions'))        
    return redirect(url_for('login')) 
    
@app.route("/delete_predictions", methods =['GET'])
def delete_predictions():
    if 'loggedin' in session:
        predictions_id = request.args.get('predictions_id') 
        
       
        insert_user_query = """
        DELETE FROM heart_disease_prediction WHERE id = %s
        """

        data = [int(predictions_id)]
        
        cursor.execute(insert_user_query, tuple(data))
        db.commit()
        return redirect(url_for('predictions'))
    return redirect(url_for('login'))




# Run the Flask app



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

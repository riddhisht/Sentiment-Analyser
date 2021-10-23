from flask import Flask,request,render_template
import numpy as np
import pickle
import pandas as pd
cv = pickle.load(open("Count","rb"))
t = pickle.load(open("Tfidf","rb"))

app= Flask(__name__)
def get_model():
    global model
    with open('isvm','rb') as f:
        model=pickle.load(f)
def preprocess(text):
    text=np.array([text])
    text=cv.transform(text)
    text=t.transform(text)
    return text
def preprocess1(text):
    text=cv.transform(text)
    text=t.transform(text)
    return text
get_model()
@app.route('/')
def hello():
    return render_template("senti_fi.html")
@app.route('/predict',methods=['POST'])
def predict():
    text1=[x for x in request.form.values()]
    print(text1[0])
    text1=preprocess(text1[0])
    result=model.predict(text1)
    x=''
    if result[0]==0:
        x='The review is negative'
        return render_template('predict2.html', prediction_text='{}'.format(x))
    else:
        x='The review is positive'
        return render_template('predict1.html', prediction_text='{}'.format(x))
    
    
@app.route('/predict1',methods=['POST'])
def predict1():
    f=request.files['data']
    text1=pd.read_csv(f)
    text1=pd.Series(text1['comments'])
    text1=preprocess1(text1)
    result=model.predict(text1)
    res=list(result)
    print(res)
    percentage=(res.count(1)/len(res))*100
    if percentage>75.0:
        return render_template('predict1.html',prediction_text='{}% comments are postive'.format(percentage))
    return render_template('predict2.html', prediction_text='{}% comments are negative'.format(100-percentage))

if __name__=='__main__':
    app.run(debug=True)

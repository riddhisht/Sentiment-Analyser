from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

cv = pickle.load(open("count_vect","rb"))
t = pickle.load(open("new_td.pickle","rb"))

app= Flask(__name__)
def get_model():
    global model
    with open('sentimental2.pkl','rb') as f:
        model=pickle.load(f)
def preprocess(text):
    text=cv.transform(text)
    text=t.transform(text)
    return text
get_model()
@app.route('/')
def hello():
    return render_template("senti1.html")
@app.route('/predict',methods=['GET','POST'])
def predict():
    f=request.files['data']
    text1=pd.read_csv(f)
    text1=pd.Series(text1['comments'])
    text1=preprocess(text1)
    result=model.predict(text1)
    res=list(result)
    print(res)
    percentage=(res.count(0)/len(res))*100
    return render_template('senti1.html', prediction_text='{}% comments are negative'.format(percentage))
    


if __name__=='__main__':
    app.run(debug=True)



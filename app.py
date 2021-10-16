import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction1 = model.predict(final_features)
    if(prediction1 == [0.]) :
        output1 = 'no'
    else :
        output1 = 'yes'
    prediction2 = model2.predict(final_features)
    if(prediction2 == [0.]) :
        output2 = 'no'
    else :
        output2 = 'yes'
    return render_template('index.html',prediction1_text ='Will it rain today ? {}'.format(output1),prediction2_text ='Will it rain tomorrow ? {}'.format(output2))

if __name__=="__main__":
    app.run(debug=True)
import numpy as np
from flask import Flask, request, render_template
from pickle import load

#initialization of Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    loaded_model = load(open('salary.pkl', 'rb'))
    prediction = loaded_model.predict(final_features)

    return render_template('index.html', prediction_text="$ {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=False)
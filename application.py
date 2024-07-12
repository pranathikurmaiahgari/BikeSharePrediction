from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
                request.form['instant'],
                request.form['dteday'],
                request.form['season'],
                request.form['yr'],
                request.form['mnth'],
                request.form['holiday'],
                request.form['weekday'],
                request.form['workingday'],
                request.form['weathersit'],
                request.form['temp'],
                request.form['atemp'],
                request.form['hum'],
                request.form['windspeed'],
                request.form['casual'],
                request.form['registered']
            )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=pred

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
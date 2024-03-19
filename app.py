import traceback
from flask import Flask, render_template, request
import os
from src.JdVVNL_Load_Forcastion.pipeline.prediction import PredictionPipeline

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    print("Training Started!")
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'GET':
        try:
            # #  reading the inputs given by the user
            # startDate = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
            # endDate = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
            # # sensor = str(request.form['sensor'])
            #
            # data = [startDate, endDate]
            # data = np.array(data).reshape(1, 2)

            obj = PredictionPipeline()
            obj.predict()
            # print(f"The Predicted data is sucessfully stored in the Database")

        except Exception as e:
            print(traceback.format_exc())
            print('The Exception message is: ', e)
            return 'something is wrong'

    # else:
        # return render_template('index.html')


if __name__ == "__main__":
    app.run()

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import tensorflow as tf
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired



# Flask 세팅
app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"
bootstrap5 = Bootstrap5(app)

# 로컬 모델 & 전처리기 로드
model = tf.keras.models.load_model("model/wildfire_predictor.h5")
preprocessor = joblib.load("model/preprocessor.pkl")

# 폼 클래스 정의
class LabForm(FlaskForm):
    longitude = StringField("Longitude (1-7)", validators=[DataRequired()])
    latitude = StringField("Latitude (1-7)", validators=[DataRequired()])
    month = StringField("Month (01-Jan ~ Dec-12)", validators=[DataRequired()])
    day = StringField("Day (00-sun ~ 06-sat, 07-hol)", validators=[DataRequired()])
    avg_temp = StringField("Average Temperature", validators=[DataRequired()])
    max_temp = StringField("Max Temperature", validators=[DataRequired()])
    max_wind_speed = StringField("Max Wind Speed", validators=[DataRequired()])
    avg_wind = StringField("Average Wind", validators=[DataRequired()])
    submit = SubmitField("Submit")

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 입력값 수집 및 DataFrame 변환
        X_test = pd.DataFrame([[
            float(form.longitude.data),
            float(form.latitude.data),
            str(form.month.data),
            str(form.day.data),
            float(form.avg_temp.data),
            float(form.max_temp.data),
            float(form.max_wind_speed.data),
            float(form.avg_wind.data),
        ]], columns=[
            "longitude", "latitude", "month", "day",
            "avg_temp", "max_temp", "max_wind_speed", "avg_wind"
        ])

        # 전처리 및 예측
        X_test_prepared = preprocessor.transform(X_test)
        log_prediction = model.predict(X_test_prepared)[0][0]
        burned_area = np.expm1(log_prediction)  # 로그 역변환

        result = round(burned_area, 2)
        return render_template("result.html", res=result)

    return render_template("prediction.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)

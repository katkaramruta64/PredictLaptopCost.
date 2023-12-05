from flask import Flask,render_template ,request
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/laptop-price")
def laptopprice():
    dataset = pd.read_csv("cleane_data.csv")
    brands = sorted(dataset["brand"].unique())
    processor_brand = sorted(dataset["processor_brand"].unique())
    processor_names = sorted(dataset["processor_name"].unique())
    processor_gnrtns = sorted(dataset["processor_gnrtn"].unique())
    ram_gb = sorted(dataset["ram_gb"].unique())

    ssds = sorted(dataset["ssd"].unique())
    hdd = sorted(dataset["hdd"].unique())
    Touchscreen = sorted(dataset["Touchscreen"].unique())

    rating = sorted(dataset["rating"].unique())

    return render_template("laptopprice.html", brands = brands, processor_brand = processor_brand , processor_names = processor_names, processor_gnrtns = processor_gnrtns ,ram_gb= ram_gb , ssds = ssds,hdd= hdd , Touchscreen=Touchscreen,rating= rating)


@app.route("/laptoppriceresult")
def laptoppriceresult():

    brand = request.args.get("brand")
    processor_brand = request.args.get("processor_brand")
    processor_name = request.args.get("processor_name")
    processor_gnrtn = request.args.get("processor_gnrtn")
    ram_gb = request.args.get("ram_gb")

    ssd = request.args.get("ssd")
    hdd = request.args.get("hdd")
    Touchscreen = request.args.get("Touchscreen")
    rating = request.args.get("rating")



    pipe = pkl.load(open('LinearRegresionModel.pkl', 'rb'))

    columns = ["brand", "processor_brand", "processor_name", "processor_gnrtn", "ram_gb","ssd","hdd","Touchscreen","rating"]
    data = np.array([brand, processor_brand, processor_name, processor_gnrtn, ram_gb , ssd , hdd, Touchscreen ,rating]).reshape(1, 9)
    myinput = pd.DataFrame(columns = columns, data = data)
    result = pipe.predict(myinput)
    return render_template("laptoppriceresult.html", brand = brand, processor_brand = processor_brand, processor_name = processor_name, processor_gnrtn = processor_gnrtn, ram_gb = ram_gb , ssd = ssd , hdd= hdd ,Touchscreen = Touchscreen ,rating = rating ,result = result)
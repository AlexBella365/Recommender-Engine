from flask import Flask, jsonify, request
from movieEngine import MovieRecommender
import sqlite3

app = Flask(__name__)

@app.route('/prediction/<int:userId>',methods=['GET'])
def get(userId):
    result = recommendation_engine.predictInterest(userId)
    return jsonify(result)

@app.route('/best',methods=['GET'])
def best():
    result = recommendation_engine.bestRatedMoviesGlobally()
    return jsonify(result)

@app.route('/add',methods=['POST'])
def add():
    userId = request.json['userId']
    movieId = request.json['movieId']
    rating = request.json['rating']
    result = recommendation_engine.addData(userId,movieId,rating)
    return jsonify(result)

if __name__ == '__main__':
    global recommendation_engine
    global database
    database = './data/RecommenderSystem.db'
    connector = sqlite3.connect(database,check_same_thread=False)
    recommendation_engine = MovieRecommender(connector)
    app.run(debug=True)
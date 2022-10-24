from flask import Flask
from flask_restful import Resource, Api
from pymongo import MongoClient
from bson import ObjectId
from bson.json_util import dumps
from dotenv import dotenv_values
import pandas as pd
from model.model import RideEvaluator

# Import environment variables
config = dotenv_values('.env')

# Setup database connection
mongo_client = MongoClient(
    config['DATABASE_URL'],
    authMechanism='MONGODB-X509',
    tls=True,
    tlsCertificateKeyFile='./certificate/atlas-admin-X509-cert.pem',
    authSource='$external'
)
database = mongo_client[config['DB_NAME']]

# Initialize Flask RSETful API
app = Flask(__name__)
api = Api(app)

# Build resources
class RideScore(Resource):
    def get(self, ride_id):
        ride_data = database['rideData'].find({'metadata.rideRecordID': ObjectId(ride_id)})
        df_ride_data = pd.DataFrame(ride_data)
        ride_evaluator = RideEvaluator(df_ride_data)
        scores = ride_evaluator.evaluate()

        return scores

api.add_resource(RideScore, '/rideScore/<string:ride_id>')

# Run app
if __name__ == '__main__':
    app.run()
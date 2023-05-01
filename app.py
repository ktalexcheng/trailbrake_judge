from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort
from waitress import serve
import os
import pandas as pd
from model.model import RideEvaluator

# Initialize Flask RSETful API
app = Flask(__name__)
api = Api(app)


class RideScore(Resource):
    def post(self):
        # Get data from REST request body
        try:
            ride_data = request.get_json()
        except Exception as err:
            return abort(400, message=str(err))

        try:
            # Parse data into dataframe
            df_ride_data = pd.DataFrame(ride_data["rideData"])
            df_ride_data["timestamp"] = pd.to_datetime(df_ride_data["timestamp"])

            # Evaluate ride
            ride_evaluator = RideEvaluator(df_ride_data)
            scores = ride_evaluator.evaluate()

            # Calculate other metrics: Duration, distance traveled, max acceleration
            total_distance = ride_evaluator.get_total_distance()
            total_duration = ride_evaluator.get_total_duration()
            max_acceleration = ride_evaluator.get_max_acceleration()
            acceleration_rms = ride_evaluator.get_acceleration_rms()

            return jsonify(
                {
                    "rideMeta": {
                        "distance": total_distance,
                        "duration": total_duration,
                        "maxAcceleration": max_acceleration,
                        "accelerationRms": acceleration_rms,
                    },
                    "rideScore": scores,
                }
            )
        except Exception as err:
            return abort(500, message=str(err))


api.add_resource(RideScore, "/rideScore")

# Run app
if __name__ == "__main__":
    # os.environ['PORT'] = '5000' # FOR DEVELOPMENT ONLY
    serve(app, listen=f'*:{os.environ.get("PORT", 8080)}')

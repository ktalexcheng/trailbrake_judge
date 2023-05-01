import numpy as np
import pandas as pd
import math
import quaternion


class Penalty:
    def __init__(self, penalty_light=-1, penalty_moderate=-3, penalty_harsh=-5):
        self.penalty_light = penalty_light
        self.penalty_moderate = penalty_moderate
        self.penalty_harsh = penalty_harsh

        self.light_violations = 0
        self.moderate_violations = 0
        self.harsh_violations = 0

    def add_light_violation(self, count=1):
        self.light_violations += count

    def add_moderate_violation(self, count=1):
        self.moderate_violations += count

    def add_harsh_violation(self, count=1):
        self.harsh_violations += count

    def total_penalty(self):
        total_penalty = (
            self.light_violations * self.penalty_light
            + self.moderate_violations * self.penalty_moderate
            + self.harsh_violations * self.penalty_harsh
        )

        return max(total_penalty, -100)


class RideEvaluator:
    # # G force thresholds for acceleration
    # ACCELERATION_MILD = -0.3
    # ACCELERATION_MODERATE = -0.4
    # ACCELERATION_HARSH = -0.5

    # # G force thresholds for cornering
    # CORNERING_MILD = 0.3
    # CORNERING_MODERATE = 0.4
    # CORNERING_HARSH = 0.5

    # # G force thresholds for braking
    # BRAKING_MILD = 0.4
    # BRAKING_MODERATE = 0.5
    # BRAKING_HARSH = 0.6

    # ISO 2631 m/s^2 thresholds for acceleration
    ISO_ACCELERATION_PT = -0.90  # Public transport
    ISO_ACCELERATION_ND = -2.00  # Normal driving
    ISO_ACCELERATION_AG = -5.08  # Aggressive driving

    # ISO 2631 m/s^2 thresholds for cornering
    ISO_CORNERING_PT = 0.9  # Public transport
    ISO_CORNERING_ND = 4.0  # Normal driving
    ISO_CORNERING_AG = 5.6  # Aggressive driving

    # ISO 2631 m/s^2 thresholds for braking
    ISO_BRAKING_PT = 0.90  # Public transport
    ISO_BRAKING_ND = 1.47  # Normal driving
    ISO_BRAKING_AG = 3.07  # Aggressive driving

    # ISO 2631 m/s^2 thresholds for vertical
    ISO_VERTICAL_PT = 0.10  # Public transport
    ISO_VERTICAL_ND = 0.10  # Normal driving
    ISO_VERTICAL_AG = 0.30  # Aggressive driving

    # Acceleration vector RMS thresholds from ISO 2631
    ACCELERATION_RMS_LV1 = 0.315
    ACCELERATION_RMS_LV2 = 0.500
    ACCELERATION_RMS_LV3 = 0.800
    ACCELERATION_RMS_LV4 = 1.250
    ACCELERATION_RMS_LV5 = 2.000

    # # Gravity constant
    # GRAVITY = 9.81

    # Penalty object for each component
    acceleration_penalty = Penalty()
    braking_penalty = Penalty()
    cornering_penalty = Penalty()

    # Metrics
    acceleration_rms = None
    max_acceleration = None
    total_duration = None
    total_distance = None

    def __init__(self, df):
        self.df_ride = df.copy()

        # Transform accelerometer data
        self.df_ride = self.df_ride.apply(self._apply_transform_accelerometer, axis=1)

        # Create model for comfort score based on acceleration vector RMS
        model_x = np.array(
            [
                0,
                self.ACCELERATION_RMS_LV1,
                self.ACCELERATION_RMS_LV2,
                self.ACCELERATION_RMS_LV3,
                self.ACCELERATION_RMS_LV4,
                self.ACCELERATION_RMS_LV5,
            ]
        )
        model_y = np.array([100, 95, 90, 80, 60, 0])
        self.model = np.poly1d(np.polyfit(model_x, model_y, 2))

        # Ride score starts at 100 with points deducted for violation
        self.comfort_score = 100
        self.acceleration_score = 100
        self.braking_score = 100
        self.cornering_score = 100
        self.ride_score = 100

    def _distance_between(self, start_lat, start_long, end_lat, end_long):
        """
        Calculates distance between two coordinates using haversine formula
        """
        if not (start_lat and start_long and end_lat and end_long):
            return 0

        distance = 0
        R = 6371000  # meters
        phi_1 = math.radians(start_lat)
        phi_2 = math.radians(end_lat)
        lambda_1 = math.radians(start_long)
        lambda_2 = math.radians(end_long)

        delta_phi = phi_2 - phi_1
        delta_lambda = lambda_2 - lambda_1

        distance = (
            2
            * R
            * math.asin(
                math.sqrt(
                    math.pow(math.sin(delta_phi / 2), 2)
                    + math.cos(phi_1)
                    * math.cos(phi_2)
                    * math.pow(math.sin(delta_lambda / 2), 2)
                )
            )
        )

        return distance

    def _inverse_rotate_vector(self, vector, q_orientation):
        """
        Inverse rotate vector data using the orientation quaternion
        """
        q_w = q_orientation[0]
        q_x = q_orientation[1]
        q_y = q_orientation[2]
        q_z = q_orientation[3]

        q = np.quaternion(q_w, q_x, q_y, q_z)
        R = quaternion.as_rotation_matrix(q)
        v = np.array(vector)
        v_inverse = np.dot(R.T, v)

        return v_inverse

    def _apply_transform_accelerometer(self, row):
        """
        Transforms accelerometer data to East-North-Up orientation. In this orientation:
            Y-axis (positive) = braking
            Y-axis (negative) = accelerating
            X-axis = cornering
            Z-axis = vertical
        """
        v = (row["accelerometerX"], row["accelerometerY"], row["accelerometerZ"])
        q = (row["rotationW"], row["rotationX"], row["rotationY"], row["rotationZ"])

        v_inverse = self._inverse_rotate_vector(v, q)
        row["_accelerometerX_t"] = v_inverse[0]
        row["_accelerometerY_t"] = v_inverse[1]
        row["_accelerometerZ_t"] = v_inverse[2]

        return row

    def _df_calc_immediate_speed(self):
        """
        Calculate immediate speed (km/hr)
        """
        # Get the first timestamp (moment of update) for each coordinate
        self.df_ride["_gps_update_ts"] = self.df_ride.groupby(
            ["locationLong", "locationLat"]
        )["timestamp"].transform("min")

        # Shift data one row down and use to find when GPS is updated
        self.df_ride["_prev_gps_update_ts"] = self.df_ride["_gps_update_ts"].shift(1)
        self.df_ride["_prev_lat"] = self.df_ride["locationLat"].shift(1)
        self.df_ride["_prev_long"] = self.df_ride["locationLong"].shift(1)

        # Calculate distance traveled and immediate speed
        self.df_ride["_distance_moved_from_last_gps"] = self.df_ride[
            ["locationLat", "locationLong", "_prev_lat", "_prev_long"]
        ].apply(
            lambda row: self._distance_between(
                row["_prev_lat"],
                row["_prev_long"],
                row["locationLat"],
                row["locationLong"],
            ),
            axis=1,
        )
        # Note: Convert m/s to km/hr --> (m/s) * (km / 1000 meter) / (hr / 3600 sec) --> 18/5
        self.df_ride["_speed_kmhr"] = (
            self.df_ride["_distance_moved_from_last_gps"]
            / (
                (self.df_ride["_gps_update_ts"] - self.df_ride["_prev_gps_update_ts"])
                / pd.Timedelta(seconds=1)
            )
            * (18 / 5)
        )
        self.df_ride["_speed_kmhr_filled"] = self.df_ride["_speed_kmhr"].ffill()

    def get_acceleration_rms(self):
        def root_mean_quadratic(series):
            return np.mean(series**4) ** (1 / 4)

        def root_mean_square(series):
            return np.sqrt(np.mean(series**2))

        def vibration_dose_value(series):
            return (series**4).sum() ** (1 / 4)

        def calc_running_rms(input_col, output_col):
            self.df_ride[output_col] = self.df_ride.rolling("1s", on="timestamp")[
                input_col
            ].apply(root_mean_square)

        def calc_running_rmq(input_col, output_col):
            self.df_ride[output_col] = self.df_ride.rolling("1s", on="timestamp")[
                input_col
            ].apply(root_mean_quadratic)

        def calc_vdv(input_col, output_col):
            self.df_ride[output_col] = (
                self.df_ride[input_col].expanding().apply(vibration_dose_value)
            )

        # calc_running_rms(input_col="_accelerometerX_t", output_col="_accelerometerX_rms")
        # calc_running_rms(input_col="_accelerometerY_t", output_col="_accelerometerY_rms")
        # calc_running_rms(input_col="_accelerometerZ_t", output_col="_accelerometerZ_rms")
        # calc_running_rmq(input_col="_accelerometerX_t", output_col="_accelerometerX_rmq")
        # calc_running_rmq(input_col="_accelerometerY_t", output_col="_accelerometerY_rmq")
        # calc_running_rmq(input_col="_accelerometerZ_t", output_col="_accelerometerZ_rmq")
        # calc_vdv(input_col="_accelerometerX_t", output_col="_accelerometerX_vdv")
        # calc_vdv(input_col="_accelerometerY_t", output_col="_accelerometerY_vdv")
        # calc_vdv(input_col="_accelerometerZ_t", output_col="_accelerometerZ_vdv")

        if self.acceleration_rms is None:
            accel_x_rms = root_mean_square(self.df_ride["_accelerometerX_t"])
            accel_y_rms = root_mean_square(self.df_ride["_accelerometerY_t"])
            accel_z_rms = root_mean_square(self.df_ride["_accelerometerZ_t"])
            self.acceleration_rms = math.sqrt(
                accel_x_rms**2 + accel_y_rms**2 + accel_z_rms**2
            )

        return self.acceleration_rms

    def get_max_acceleration(self):
        if self.max_acceleration is None:
            # Calculate absolute acceleration
            self.df_ride["_abs_acceleration"] = np.sqrt(
                sum(
                    [
                        self.df_ride["accelerometerX"] ** 2,
                        self.df_ride["accelerometerY"] ** 2,
                        self.df_ride["accelerometerZ"] ** 2,
                    ]
                )
            )

            self.max_acceleration = self.df_ride["_abs_acceleration"].max()

        return self.max_acceleration

    def get_total_duration(self):
        if self.total_duration is None:
            self.total_duration = (
                self.df_ride["timestamp"].max() - self.df_ride["timestamp"].min()
            ).total_seconds()

        return self.total_duration

    def get_total_distance(self):
        if self.total_distance is None:
            if "_distance_moved" not in self.df_ride:
                for i in range(1, len(self.df_ride)):
                    self.df_ride.loc[i, "_distance_moved"] = self._distance_between(
                        start_lat=self.df_ride.loc[i - 1, "locationLat"],
                        start_long=self.df_ride.loc[i - 1, "locationLong"],
                        end_lat=self.df_ride.loc[i, "locationLat"],
                        end_long=self.df_ride.loc[i, "locationLong"],
                    )

            self.total_distance = self.df_ride["_distance_moved"].sum()

        return self.total_distance

    def evaluate(self):
        # Calculate distance traveled
        self.get_total_distance()

        # Calculate immediate speed
        self._df_calc_immediate_speed()

        # Calculate acceleration rms
        self.get_acceleration_rms()

        self.comfort_score = round(max(self.model(self.acceleration_rms), 0), 0)

        # Group by and aggregate metrics
        self.df_grouped = self.df_ride
        self.df_grouped = (
            self.df_grouped.assign(
                timestamp_3s=self.df_grouped["timestamp"].dt.floor(freq="3s")
            )
            .groupby(by=["timestamp_3s"])[
                [
                    "_accelerometerX_t",
                    "_accelerometerY_t",
                    "_accelerometerZ_t",
                    "locationLat",
                    "locationLong",
                    "_speed_kmhr_filled",
                ]
            ]
            .agg(
                {
                    "_accelerometerX_t": ["min", "max"],
                    "_accelerometerY_t": ["min", "max"],
                    "_accelerometerZ_t": ["min", "max"],
                    "locationLat": ["mean"],
                    "locationLong": ["mean"],
                    "_speed_kmhr_filled": ["min", "max"],
                }
            )
        )

        for _, row in self.df_grouped.iterrows():
            # Positive-Y G = braking
            _accel_max = row.loc[("_accelerometerY_t", "max")]
            # Negative-Y G = acceleration
            _accel_min = row.loc[("_accelerometerY_t", "min")]
            _corner_right = row.loc[("_accelerometerX_t", "max")]
            _corner_left = row.loc[("_accelerometerX_t", "min")]
            _corner_max = max(abs(_corner_left), abs(_corner_right))

            # # Positive-Y G = braking
            # _accel_max_G = _accel_max / self.GRAVITY
            # # Negative-Y G = acceleration
            # _accel_min_G = _accel_min / self.GRAVITY
            # _corner_G = _corner / self.GRAVITY

            # # Flag penalties for acceleration
            # if self.ACCELERATION_MILD > _accel_min_G > self.ACCELERATION_MODERATE:
            #     self.acceleration_penalty.add_light_violation()
            # if self.ACCELERATION_MODERATE > _accel_min_G > self.ACCELERATION_HARSH:
            #     self.acceleration_penalty.add_moderate_violation()
            # if self.ACCELERATION_HARSH > _accel_min_G:
            #     self.acceleration_penalty.add_harsh_violation()

            # # Flag penalties for braking
            # if self.BRAKING_MILD < _accel_max_G < self.BRAKING_MODERATE:
            #     self.braking_penalty.add_light_violation()
            # if self.BRAKING_MODERATE < _accel_max_G < self.BRAKING_HARSH:
            #     self.braking_penalty.add_moderate_violation()
            # if self.BRAKING_HARSH < _accel_max_G:
            #     self.braking_penalty.add_harsh_violation()

            # # Flag penalties for cornering
            # if self.CORNERING_MILD < _corner_G < self.CORNERING_MODERATE:
            #     self.cornering_penalty.add_light_violation()
            # if self.CORNERING_MODERATE < _corner_G < self.CORNERING_HARSH:
            #     self.cornering_penalty.add_moderate_violation()
            # if self.CORNERING_HARSH < _corner_G:
            #     self.cornering_penalty.add_harsh_violation()

            # Flag penalties for acceleration
            if self.ISO_ACCELERATION_AG > _accel_min:
                self.acceleration_penalty.add_harsh_violation()
            elif self.ISO_ACCELERATION_ND > _accel_min:
                self.acceleration_penalty.add_moderate_violation()
            elif self.ISO_ACCELERATION_PT > _accel_min:
                self.acceleration_penalty.add_light_violation()

            # Flag penalties for braking
            if self.ISO_BRAKING_AG < _accel_max:
                self.braking_penalty.add_harsh_violation()
            elif self.ISO_BRAKING_ND < _accel_max:
                self.braking_penalty.add_moderate_violation()
            elif self.ISO_BRAKING_PT < _accel_max:
                self.braking_penalty.add_light_violation()

            # Flag penalties for cornering
            if self.ISO_CORNERING_AG < _corner_max:
                self.cornering_penalty.add_harsh_violation()
            elif self.ISO_CORNERING_ND < _corner_max:
                self.cornering_penalty.add_moderate_violation()
            elif self.ISO_CORNERING_PT < _corner_max:
                self.cornering_penalty.add_light_violation()

        self.acceleration_score += self.acceleration_penalty.total_penalty()
        self.braking_score += self.braking_penalty.total_penalty()
        self.cornering_score += self.cornering_penalty.total_penalty()
        self.ride_score = (
            sum(
                [
                    self.comfort_score,
                    self.acceleration_score,
                    self.braking_score,
                    self.cornering_score,
                ]
            )
            // 4
        )

        # print(f'Speed score: {self.speed_score}')
        # print(f'Acceleration score: {self.acceleration_score}')
        # print(f'Braking score: {self.braking_score}')
        # print(f'Cornering score: {self.cornering_score}')
        # print(f'Ride score: {self.ride_score}')

        score_breakdown = {
            "overall": self.ride_score,
            "comfort": self.comfort_score,
            "acceleration": self.acceleration_score,
            "braking": self.braking_score,
            "cornering": self.cornering_score,
        }

        return score_breakdown

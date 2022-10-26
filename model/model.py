class Penalty:
    def __init__(self, penalty_light=-1, penalty_moderate=-3, penalty_harsh=-5):
        self.penalty_light = penalty_light
        self.penalty_moderate = penalty_moderate
        self.penalty_harsh = penalty_harsh

        self.light_violations = 0
        self.moderate_violations = 0
        self.harsh_violations = 0

    def add_light_violation(self):
        self.light_violations += 1

    def add_moderate_violation(self):
        self.moderate_violations += 1

    def add_harsh_violation(self):
        self.harsh_violations += 1

    def total_penalty(self):
        total_penalty = self.light_violations * self.penalty_light + \
            self.moderate_violations * self.penalty_moderate + \
            self.harsh_violations * self.penalty_harsh

        return max(total_penalty, -100)

class RideEvaluator:
    # G force thresholds for acceleration
    ACCELERATION_MILD = -0.3
    ACCELERATION_MODERATE = -0.4
    ACCELERATION_HARSH = -0.5

    # G force thresholds for cornering
    CORNERING_MILD = 0.3
    CORNERING_MODERATE = 0.4
    CORNERING_HARSH = 0.5

    # G force thresholds for braking
    BRAKING_MILD = 0.4
    BRAKING_MODERATE = 0.5
    BRAKING_HARSH = 0.6

    # Gravity constant
    GRAVITY = 9.81

    # Penalty object for each component
    speed_penalty = Penalty()
    acceleration_penalty = Penalty()
    braking_penalty = Penalty()
    cornering_penalty = Penalty()
    
    def __init__(self, df):
        self.df_original = df

        # Ride score starts at 100 with points deducted for violation
        self.speed_score = 100
        self.acceleration_score = 100
        self.braking_score = 100
        self.cornering_score = 100
        self.ride_score = 100

    def _distance_between(self, start_lat, start_long, end_lat, end_long):
        '''
        Calculates distance between two coordinates using haversine formula
        '''
        import math

        distance = 0
        R = 6371000 # meters
        phi_1 = math.radians(start_lat)
        phi_2 = math.radians(end_lat)
        lambda_1 = math.radians(start_long)
        lambda_2 = math.radians(end_long)

        delta_phi = phi_2 - phi_1
        delta_lambda = lambda_2 - lambda_1
        
        distance = 2 * R * math.asin(
            math.sqrt(
                math.pow(math.sin(delta_phi / 2), 2)
                    + math.cos(phi_1) * math.cos(phi_2) * math.pow(math.sin(delta_lambda / 2), 2)
            )
        )

        return distance

    def max_acceleration(self):
        import numpy as np

        _df = self.df_original

        # Calculate absolute acceleration
        _df['_abs_acceleration'] = np.sqrt(
            sum([
                _df['accelerometerX'] ** 2, 
                _df['accelerometerY'] ** 2, 
                _df['accelerometerZ'] ** 2
            ])
        )

        return _df['_abs_acceleration'].max()

    def total_duration(self):
        return (self.df_original['timestamp'].max() - self.df_original['timestamp'].min()).total_seconds()

    def total_distance(self):
        _df = self.df_original

        for i in range(1, len(_df)):
            _df.loc[i, '_distance_moved'] = self._distance_between(
                start_lat=_df.loc[i - 1, 'locationLat'],
                start_long=_df.loc[i - 1, 'locationLong'],
                end_lat=_df.loc[i, 'locationLat'],
                end_long=_df.loc[i, 'locationLong'],
            )

        return _df['_distance_moved'].sum()

    def evaluate(self):
        self.df_processed = self.df_original.assign(
            timestamp_s=self.df_original['timestamp'].dt.floor(freq='s')
        ).groupby(by=['timestamp_s'])[['accelerometerX', 'accelerometerY', 'accelerometerZ']].agg({
            'accelerometerX': ['min', 'max'],
            'accelerometerY': ['min', 'max'],
            'accelerometerZ': ['min', 'max']
        })

        for i, row in self.df_processed.iterrows():
            _accelX_max_G = abs(row.loc[('accelerometerX', 'max')] / self.GRAVITY)
            # _accelY_max_G = abs(row.loc[('accelerometerY', 'max')] / self.GRAVITY)
            _accelZ_max_G = abs(row.loc[('accelerometerZ', 'max')] / self.GRAVITY)

            _accelX_min_G = abs(row.loc[('accelerometerX', 'min')] / self.GRAVITY)
            # _accelY_min_G = abs(row.loc[('accelerometerY', 'min')] / self.GRAVITY)
            _accelZ_min_G = abs(row.loc[('accelerometerZ', 'min')] / self.GRAVITY)

            _corner_G = max(abs(_accelX_min_G), abs(_accelX_max_G))

            if self.ACCELERATION_MILD > _accelZ_min_G > self.ACCELERATION_MODERATE: self.acceleration_penalty.add_light_violation()
            if self.ACCELERATION_MODERATE > _accelZ_min_G > self.ACCELERATION_HARSH: self.acceleration_penalty.add_moderate_violation()
            if self.ACCELERATION_HARSH > _accelZ_min_G: self.acceleration_penalty.add_harsh_violation()

            if self.BRAKING_MILD < _accelZ_max_G < self.BRAKING_MODERATE: self.braking_penalty.add_light_violation()
            if self.BRAKING_MODERATE < _accelZ_max_G < self.BRAKING_HARSH: self.braking_penalty.add_moderate_violation()
            if self.BRAKING_HARSH < _accelZ_max_G: self.braking_penalty.add_harsh_violation()

            if self.CORNERING_MILD < _corner_G < self.CORNERING_MODERATE: self.cornering_penalty.add_light_violation()
            if self.CORNERING_MODERATE < _corner_G < self.CORNERING_HARSH: self.cornering_penalty.add_moderate_violation()
            if self.CORNERING_HARSH < _corner_G: self.cornering_penalty.add_harsh_violation()

        self.speed_score += self.speed_penalty.total_penalty()
        self.acceleration_score += self.acceleration_penalty.total_penalty()
        self.braking_score += self.braking_penalty.total_penalty()
        self.cornering_score += self.cornering_penalty.total_penalty()
        self.ride_score = sum([self.speed_score, self.acceleration_score, self.braking_score, self.cornering_score]) // 4

        # print(f'Speed score: {self.speed_score}')
        # print(f'Acceleration score: {self.acceleration_score}')
        # print(f'Braking score: {self.braking_score}')
        # print(f'Cornering score: {self.cornering_score}')
        # print(f'Ride score: {self.ride_score}')

        score_breakdown = {
            'overall': self.ride_score,
            'speed': self.speed_score,
            'acceleration': self.acceleration_score,
            'braking': self.braking_score,
            'cornering': self.cornering_score,
        }

        return score_breakdown
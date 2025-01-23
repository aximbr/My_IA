from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data

boston_dataset = fetch_openml(name='boston', version=1)
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

features['CHAS'] = features['CHAS'].astype('float64')
features['RAD'] = features['RAD'].astype('float64')

log_prices = np.log(boston_dataset.target).to_numpy()
# add to_numpy() method to force return a Array otherwise returns a serie

target = pd.DataFrame(log_prices, columns=['PRICE'])

#mean values
property_stats = features.mean().to_frame().transpose()

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)


MSE = mean_squared_error(target, fitted_vals )
RMSE = np.sqrt(MSE)

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE /np.median(boston_dataset.target)


def get_log_estimate(nr_rooms, students_per_classroom, 
                     next_to_river=False, 
                     high_confidence=False):
    # Configure Property
    property_stats['RM'] = nr_rooms
    property_stats['PTRATIO'] = students_per_classroom
    if next_to_river:
        property_stats['CHAS'] = 1
    else:
        property_stats['CHAS'] = 0

    # Make prediction
    log_estimate = regr.predict(property_stats)

    # Calc range
    if high_confidence:
        # calc 2 ST. D
        upper_bound = log_estimate + 2* RMSE
        lower_bound = log_estimate - 2* RMSE
        interval = 95
    else:
        # calc 1 ST. D.
        # calc 2 ST. D
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, range=True):
    """Estimate the price of a property in Boston.

    Keyword arguments:
    rm -- number of rooms int the property.
    ptratio -- number of students per teacher in the classroom for the school in the area.
    chas -- True if the property is next to the river, False otherwise.
    large_range -- True for a 95% prediction interval, False for a 68% interval.

    """

    if rm < 1 or ptratio < 1 :
        print('That is unrealistic! Try again.')
        return
    

    log_est, upper, lower, conf = get_log_estimate(rm, ptratio,
                                                   next_to_river=chas,
                                                    high_confidence=range )

    # convert to today's dollar
    dollar_est = np.e**(log_est) * 1000 * SCALE_FACTOR
    dollar_up = np.e**(upper) * 1000 * SCALE_FACTOR
    dollar_low = np.e**(lower) * 1000 * SCALE_FACTOR

    rounded_est = np.around(dollar_est,-3)
    rounded_up = np.around(dollar_up,-3)
    rounded_low = np.around(dollar_low,-3)

    print(f'The estimate property value is USD {rounded_est}')
    print(f'At {conf}% confidence of valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_up} at higher end')

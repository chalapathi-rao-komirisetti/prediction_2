from werkzeug.wrappers import Request, Response
from flask import Flask , render_template, request

app = Flask(__name__)

from geopy.geocoders import Nominatim

def Lati_longi(add1):
    geolocator = Nominatim(user_agent="AmanTiwari")
    location = geolocator.geocode(add1)
    return [location.latitude, location.longitude]


def getDistanceLatLonInKM(pick_lat, pick_lon, drop_lat, drop_lon):
    import numpy as np

    R = 6373.0

    pickup_lat = np.radians(pick_lat)
    pickup_lon = np.radians(pick_lon)
    dropoff_lat = np.radians(drop_lat)
    dropoff_lon = np.radians(drop_lon)

    dist_lon = dropoff_lon - pickup_lon
    dist_lat = dropoff_lat - pickup_lat

    a = (np.sin(dist_lat / 2)) ** 2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (np.sin(dist_lon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    return d


def time_estimate(dist):
    speed = 35  # speed in km/hours

    time = round(dist / speed, 3)

    if (time < 1):
        time = str(int(round(time * 60, 0)))
        time = time + ' min'

    elif len(str(time)) == 3 and int(str(time)[0]) > 1:
        time = str(int(time)) + ' Hours'

    elif len(str(time)) == 3 and int(str(time)[0]) == 1:
        time = str(int(time)) + ' Hour'

    else:
        time = str(round(time, 2))
        time = (time[0] + ' Hour ') + (str(round((float(time[1:])) * 60)) + ' min')

    return time

import pickle
import numpy as np

with open(
        'NewYork_taxi_price_prediction_model.pickle',
        'rb') as f:
    model = pickle.load(f)


@app.route("/", methods=['GET', 'POST'])
def my_page():
    if request.method == 'POST':

        source_location = request.form['so_loc']
        Destination_location = request.form['De_loc']
        date_time = request.form['date2time']

        source_lati = Lati_longi(source_location)[0]
        source_longi = Lati_longi(source_location)[1]

        Destination_lati = Lati_longi(Destination_location)[0]
        Destination_longi = Lati_longi(Destination_location)[1]

        year = date_time[:4]
        month = date_time[5:7]
        day = date_time[8:10]
        Hours = date_time[11:13]
        minutes = date_time[14:]

        distance = getDistanceLatLonInKM(source_lati, source_longi, Destination_lati, Destination_longi)

        distance2 = distance

        time = time_estimate(distance)

        input_model = np.array([1, source_lati, source_longi, Destination_lati, Destination_longi,
                                year, month, day, Hours, minutes, distance],dtype=float)

        input_model = input_model.reshape(1, -1)

        print(input_model)
        total_fare = model.predict(input_model)

        print(total_fare)

        total_fare = total_fare.item(1)

        print(total_fare)
        print(source_lati, source_longi, Destination_lati, Destination_longi,
              year, month, day, Hours, minutes, distance)

        return render_template('index.html', total_fare=total_fare, total_distance=str(round(distance2, 1)) + ' KM',
                               total_time=time)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    
    app.run(debug=True)
#we created api key in aviation stack so to fetch that latest flight data we need to write some code in our file 
import os
import requests
from dotenv import load_dotenv   #to load api key in memory

load_dotenv()  #api key is loaded into memory

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

def search_flights(query):  #query is user's input 

    url = "http://api.aviationstack.com/v1/flights"  #api's url from where we get flight's data

    params = {
        "access_key": API_KEY, #providing api key
        "limit": 5   #we need 5 results at once
    }

    response = requests.get(url, params=params)  #response is generated , in that url , we give this parameter and we get response 

    data = response.json()  #response is stored in json format under the data variable 

    flights = []  #blank list which stores flight details in flights list

    if "data" in data:

        for flight in data["data"][:5]:

            airline = flight.get("airline", {}).get("name", "Unknown")

            departure = flight.get(
                "departure", {}
            ).get("airport", "Unknown")

            arrival = flight.get(
                "arrival", {}
            ).get("airport", "Unknown")  #which airport has arrival

            status = flight.get("flight_status", "Unknown")

            flights.append(
                f"""
Airline: {airline}
Departure: {departure}
Arrival: {arrival}
Status: {status}
"""
            )  #appending flight list with these 

    return "\n".join(flights)

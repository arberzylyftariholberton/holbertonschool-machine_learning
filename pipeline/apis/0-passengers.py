#!/usr/bin/env python3
""" A script that uses SWAP API to create a method that
    returns the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    A function that returns a list of ships that
    can hold a given number of passengers
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0')

            try:
                # Removing comas, converting into int
                passengers_int = int(passengers.replace(",", ""))
            except ValueError:
                continue

            if passengers_int >= passengerCount:
                ships.append(ship["name"])

        # Pagination handlind
        url = data.get("next")

    return ships

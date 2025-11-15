#!/usr/bin/env python3
"""
A script that returns the list of names
of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    Returns a sorted list of the names of the home
    planets of all sentient species using the SWAPI API.
    """
    url = "https://swapi.dev/api/species/"
    home_planets = []
    seen = set()

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get('results', []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")

                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    planet_data = planet_response.json()
                    name = planet_data.get("name", "unknown")
                else:
                    name = "unknown"

                if name not in seen:
                    home_planets.append(name)
                    seen.add(name)

        url = data.get("next")

    return home_planets

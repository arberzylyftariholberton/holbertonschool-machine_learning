#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket using SpaceX API.
"""
import requests


if __name__ == "__main__":
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches"
    ).json()

    counts = {}
    for launch in launches:
        rocket_id = launch["rocket"]
        counts[rocket_id] = counts.get(rocket_id, 0) + 1

    rockets = {}
    for rocket_id in counts:
        rocket_data = requests.get(
            f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        ).json()
        rockets[rocket_id] = rocket_data.get("name", "Unknown")

    # Tuples list
    launch_list = [
        (rockets[rid], count) for rid, count in counts.items()
    ]

    def sort_key(item):
        """Return a tuple for sorting: (-count, name)"""
        name, count = item
        return -count, name

    launch_list.sort(key=sort_key)

    for rocket_name, count in launch_list:
        print(f"{rocket_name}: {count}")

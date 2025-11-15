#!/usr/bin/env python3
"""
A script that displays the first SpaceX launch with details.
"""
import requests


def first_launch():
    """
    A function that returns the first SpaceX launch
    """

    launches = requests.get(
        "https://api.spacexdata.com/v4/launches"
    ).json()

    launches.sort(key=lambda l: l.get("date_unix", float("inf")))

    first = launches[0]

    return first


if __name__ == "__main__":
    launch = first_launch()

    rocket_id = launch["rocket"]
    rocket = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    ).json()

    launchpad_id = launch["launchpad"]
    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    ).json()

    launch_name = launch.get("name")
    launch_date = launch.get("date_local")
    rocket_name = rocket.get("name")
    launchpad_name = launchpad.get("name")
    launchpad_locality = launchpad.get("locality")

    print(
        f"{launch_name} ({launch_date}) "
        f"{rocket_name} - {launchpad_name} "
        f"({launchpad_locality})"
    )

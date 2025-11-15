#!/usr/bin/env python3
"""
A script that prints the location of a GitHub user given the full API URL
"""
import sys
import time
import math
import requests


def print_user_location(api_url):
    """
    Fetching the GitHub user at api_urland print location or proper message
    """

    try:
        response = requests.get(api_url)
    except requests.RequestException:
        return

    status = response.status_code

    if status == 200:
        try:
            data = response.json()
        except ValueError:
            return

        print(data.get("location"))

    elif status == 404:
        print("Not found")

    elif status == 403:
        reset_header = response.headers.get("X-RateLimit-Reset")
        if reset_header is None:
            print("Reset in 0 min")
            return
        try:
            reset_ts = int(reset_header)
        except (TypeError, ValueError):
            print("Reset in 0 min")
            return

        now = time.time()
        seconds_left = reset_ts - now
        if seconds_left <= 0:
            mins = 0
        else:
            mins = math.ceil(seconds_left / 60.0)
        print(f"Reset in {mins} min")
    else:
        return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(0)

    api_url = sys.argv[1]
    print_user_location(api_url)

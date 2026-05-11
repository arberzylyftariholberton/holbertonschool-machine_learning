# APIs

REST API consumption tasks using the Star Wars API (SWAPI) and SpaceX API.

## Tasks

| File | Function / Script | Description |
|------|-------------------|-------------|
| `0-passengers.py` | `availableShips(passengerCount)` | Returns a list of SWAPI starships that can hold at least `passengerCount` passengers |
| `1-sentience.py` | `sentientPlanets()` | Returns the home planets of all sentient species from SWAPI |
| `2-user_location.py` | script | Prints the location of a GitHub user from the GitHub API; handles 404 and rate-limit responses |
| `3-first_launch.py` | `first_launch()` / script | Finds and displays the next upcoming SpaceX launch with rocket and launchpad details |
| `4-rocket_frequency.py` | script | Fetches all SpaceX launches and prints launch counts per rocket, sorted by count descending |

## APIs Used

- **SWAPI** — `https://swapi.dev/api/` — Star Wars universe data
- **SpaceX** — `https://api.spacexdata.com/v4/` — Launch, rocket, and launchpad data
- **GitHub** — `https://api.github.com/` — User location lookup

## Requirements

- Python 3.x
- requests

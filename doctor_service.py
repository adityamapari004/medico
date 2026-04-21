import requests

def search_doctors(specialty: str, location: str, limit: int = 5):
    """
    Search doctors using OpenStreetMap (Nominatim API)
    """

    url = "https://nominatim.openstreetmap.org/search"

    query = f"{specialty} doctor in {location}",
    query = f"{specialty} clinic {location}",
    query = f"hospital {location}"

    params = {
        "q": query,
        "format": "json",
        "limit": limit
    }

    headers = {
        "User-Agent": "doctor-finder-app"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        return []

    data = response.json()

    doctors = []

    for place in data:
        doctors.append({
            "name": place.get("display_name", "N/A").split(",")[0],
            "full_address": place.get("display_name", "N/A"),
            "latitude": float(place.get("lat")),
            "longitude": float(place.get("lon")),
            "type": place.get("type")
        })

    return doctors
import requests

def search_doctors(specialty: str, location: str, limit: int = 20):
    """
    Search for doctors/clinics/hospitals using OpenStreetMap Nominatim API.
    Tries multiple query strategies to get the best results.
    """

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "mediguide-doctor-finder/1.0"}

    # Strategy 1: specialty + doctor + location
    # Strategy 2: specialty + clinic + location  
    # Strategy 3: hospital + location (broad fallback)
    queries = [
        f"{specialty} doctor {location}",
        f"{specialty} clinic {location}",
        f"{specialty} hospital {location}",
        f"hospital {location}",
    ]

    seen_names = set()
    doctors = []

    for query in queries:
        if len(doctors) >= limit:
            break

        params = {
            "q": query,
            "format": "json",
            "limit": limit,
            "addressdetails": 1,
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=8)
            if response.status_code != 200:
                continue

            for place in response.json():
                if len(doctors) >= limit:
                    break

                name = place.get("display_name", "N/A").split(",")[0].strip()

                # De-duplicate
                if name in seen_names:
                    continue
                seen_names.add(name)

                doctors.append({
                    "name": name,
                    "full_address": place.get("display_name", "N/A"),
                    "latitude": float(place.get("lat", 0)),
                    "longitude": float(place.get("lon", 0)),
                    "type": place.get("type", "N/A"),
                    "specialty": specialty,
                })

        except Exception:
            continue

    return doctors
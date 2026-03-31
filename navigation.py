import requests
import pyttsx3
import math

# -------------------------
# TTS Setup
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------------
# Get Current Location (via IP API)
# -------------------------
def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        if response.status_code == 200:
            data = response.json()
            return data["lat"], data["lon"]
        else:
            return None, None
    except Exception as e:
        print("Error fetching location:", e)
        return None, None

# -------------------------
# Haversine Distance Calculation
# -------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# -------------------------
# Search Nearby Places with Nominatim
# -------------------------
def search_places(query, lat, lon, limit=10):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
        # Use a bounding box around the current location (roughly 0.1 degree)
        "viewbox": f"{lon-0.1},{lat+0.1},{lon+0.1},{lat-0.1}",
        "bounded": 1
    }
    headers = {"User-Agent": "NavigationAssistant/1.0"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error in Nominatim search:", response.status_code)
        return []

# -------------------------
# Get Walking Directions using OSRM
# -------------------------
def get_walking_directions(start_lat, start_lon, end_lat, end_lon):
    url = f"http://router.project-osrm.org/route/v1/walking/{start_lon},{start_lat};{end_lon},{end_lat}?overview=false&steps=true"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error from OSRM:", response.status_code)
        return None, None
    data = response.json()
    if data.get("code") != "Ok":
        print("OSRM error:", data.get("message", ""))
        return None, None

    # Extract overall distance and duration
    route = data["routes"][0]
    total_distance = route["distance"]  # in meters
    total_duration = route["duration"]  # in seconds
    steps = route["legs"][0]["steps"]

    # Build natural language instructions
    instructions = []
    for step in steps:
        maneuver = step.get("maneuver", {})
        m_type = maneuver.get("type", "")
        m_modifier = maneuver.get("modifier", "")
        street = step.get("name", "")
        distance = step.get("distance", 0)
        # Generate instruction based on maneuver type
        if m_type == "depart":
            instr = f"Start on {street}" if street else "Start"
        elif m_type == "arrive":
            instr = "Arrive at your destination"
        elif m_type == "turn":
            instr = f"Turn {m_modifier} onto {street}" if street else f"Turn {m_modifier}"
        elif m_type == "continue":
            instr = f"Continue straight on {street}" if street else "Continue straight"
        elif m_type == "new name":
            instr = f"Continue onto {street}" if street else "Continue"
        elif m_type == "roundabout":
            exit_num = maneuver.get("exit", "")
            instr = f"Enter the roundabout and take exit {exit_num}" if exit_num else "Enter the roundabout"
        else:
            instr = m_type.capitalize()
        instructions.append(f"{instr} for {distance:.1f} meters")
    return instructions, total_distance, total_duration

def main():
    print("Navigation Assistant")
    
    # 1. Get current location
    current_lat, current_lon = get_current_location()
    if current_lat is None:
        print("Unable to determine location automatically.")
        speak("Unable to determine your location automatically. Please enter your location manually.")
        loc_input = input("Enter your current location (lat,lon): ")
        try:
            current_lat, current_lon = map(float, loc_input.split(","))
        except:
            print("Invalid format. Exiting.")
            return
    print(f"Your current location: {current_lat:.4f}, {current_lon:.4f}")
    
    # 2. Do not speak your location; instead, just display it.
    print(f"Your current location is approximately: {current_lat:.4f}, {current_lon:.4f}")
    
    # 3. Ask for destination query
    speak("Where do you want to go?")
    query = input("Enter destination query (e.g., hospital, school): ").strip()
    if not query:
        print("No query provided. Exiting.")
        speak("No query provided.")
        return
    
    # 4. Search for places near current location
    places = search_places(query, current_lat, current_lon, limit=10)
    if not places:
        print("No places found for the query.")
        speak("No places found for your query.")
        return
    
    # 5. Compute straight-line distance for each result and sort by distance
    for place in places:
        try:
            p_lat = float(place["lat"])
            p_lon = float(place["lon"])
            place["distance"] = haversine_distance(current_lat, current_lon, p_lat, p_lon)
        except:
            place["distance"] = float('inf')
    places = sorted(places, key=lambda x: x["distance"])
    
    # 6. Display the options (only name and distance) – do not show full address
    print("\nFound the following options:")
    options = []
    for idx, place in enumerate(places):
        # Use the first part of display_name as the name
        full_name = place.get("display_name", "Unknown")
        name = full_name.split(",")[0]
        distance = place.get("distance", 0)
        # Convert distance to km for display
        print(f"{idx+1}. {name} - {distance/1000:.2f} km away")
        options.append(place)
    
    speak("Please choose your destination by entering the corresponding number.")
    selection = input("Select a destination (number): ").strip()
    try:
        sel_idx = int(selection) - 1
        if sel_idx < 0 or sel_idx >= len(options):
            print("Invalid selection.")
            speak("Invalid selection.")
            return
    except:
        print("Invalid input.")
        speak("Invalid input.")
        return
    
    selected = options[sel_idx]
    # For display, show the full address, distance and approximate travel time
    dest_name = selected.get("display_name", "Unknown destination")
    dest_lat = float(selected["lat"])
    dest_lon = float(selected["lon"])
    distance_to_dest = selected.get("distance", 0)
    
    # Get walking directions via OSRM
    directions, total_distance, total_duration = get_walking_directions(current_lat, current_lon, dest_lat, dest_lon)
    if directions is None:
        print("Unable to retrieve walking directions.")
        speak("Unable to retrieve walking directions.")
        return
    
    # Convert total duration to minutes
    duration_minutes = total_duration / 60.0
    print("\nDestination Details:")
    print(f"Destination: {dest_name}")
    print(f"Distance: {distance_to_dest/1000:.2f} km away")
    print(f"Estimated travel time: {duration_minutes:.0f} minutes")
    
    # Construct the full message with detailed directions:
    # We'll try to make it more natural by prepending with "Start walking" and ending with "Arrive at your destination."
    natural_instructions = []
    if directions:
        # OSRM may return a series of instructions like "Start for 226.6 meters" etc.
        # We can optionally refine them. For now, we assume directions is a list of strings.
        natural_instructions.append("Start walking")
        natural_instructions.extend(directions)
        natural_instructions.append("Arrive at your destination")
    final_message = f"Destination: {dest_name}. It is {distance_to_dest/1000:.2f} kilometers away and will take approximately {duration_minutes:.0f} minutes to walk. " \
                    + ". ".join(natural_instructions) + "."
    print("\nWalking Directions:")
    print(final_message)
    speak(final_message)

if __name__ == "__main__":
    main()

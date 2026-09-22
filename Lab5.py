import streamlit as st
import requests
import json
from openai import OpenAI

#### GET THE WEATHER ####
# location can be a city, a zip code, an airport code ('SYR'),
# or a landmark ('Eiffel+Tower')
# note: hard codes units to degrees Fahrenheit
def get_current_weather(location):
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')

    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f'Could not find a location named {location}')

    # j1 has three top-level sections:
    #   current_condition -- one entry, conditions right now
    #   weather           -- three entries, one per day, each with
    #                        min/max, astronomy, and hourly forecasts
    #   nearest_area      -- the location wttr.in actually matched
    current = data['current_condition'][0]
    today = data['weather'][0]
    astronomy = today['astronomy'][0]
    area = data['nearest_area'][0]

    # wttr.in pads some description strings, so strip them
    matched = ', '.join(
        part[0]['value'] for part in
        (area['areaName'], area['region'], area['country'])
        if part[0]['value'].strip()
    )

    # Conditions through the rest of the day drive the advice as much as
    # the current reading does - a warm afternoon can turn cold by dinner
    hourly = []
    for slot in today['hourly']:
        hour = int(slot['time']) // 100
        hourly.append({
            'time': f'{hour:02d}:00',
            'temperature': float(slot['tempF']),
            'feels_like': float(slot['FeelsLikeF']),
            'description': slot['weatherDesc'][0]['value'].strip(),
            'chance_of_rain': int(slot['chanceofrain'])
        })

    return {'location': location,
            'matched_location': matched,
            'temperature': float(current['temp_F']),
            'feels_like': float(current['FeelsLikeF']),
            'description': current['weatherDesc'][0]['value'].strip(),
            'wind_mph': float(current['windspeedMiles']),
            'wind_direction': current['winddir16Point'],
            'humidity': int(current['humidity']),
            'precipitation_inches': float(current['precipInches']),
            'high_today': float(today['maxtempF']),
            'low_today': float(today['mintempF']),
            'sunrise': astronomy['sunrise'],
            'sunset': astronomy['sunset'],
            'hourly': hourly
            }

DEFAULT_LOCATION = 'Syracuse, NY'

#### THE TOOL DEFINITION ####
# This is all the model sees. It never reads the Python function, so the
# descriptions here are what it uses to decide whether and how to call it.
weather_tool = {
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get current conditions and the rest of today\'s '
                       'hourly forecast for a location. Use this before '
                       'giving any advice about clothing or outdoor plans.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'City and state or country, e.g. '
                                   '"Syracuse, NY" or "Lima, Peru". Omit '
                                   'if the user did not name a place.'
                }
            }
        }
    }
}

SYSTEM_PROMPT = (
    'You are a "what to wear" assistant. Given a location, look up the '
    'weather and then tell the user what to wear today and suggest outdoor '
    'activities that suit the conditions.\n\n'
    'Do not guess at the weather - always use the get_current_weather tool.\n\n'
    'Write the answer as a single self-contained response:\n'
    '- Do not mention the lookup, the tool, or the data you were given.\n'
    '- Note how conditions change through the day when it affects the '
    'advice, and tie activity suggestions to the times that suit them.\n'
    '- Do not ask the user follow-up questions. They cannot reply.\n'
    '- Keep it brief - a short line on conditions, then clothing, then '
    'two or three activities.'
)


#### RUN THE REQUESTED TOOL ####
# The model asked for weather; this runs the actual function and hands the
# result back in the shape the API expects for a tool response.
def run_weather_tool(call, messages, reply):
    args = json.loads(call.function.arguments)

    # The model may ask for weather without naming a place
    location = args.get('location') or DEFAULT_LOCATION

    weather = get_current_weather(location)

    # A 'tool' message must follow the assistant message that requested it
    messages.append(reply.to_dict())
    messages.append({
        'role': 'tool',
        'tool_call_id': call.id,
        'name': call.function.name,
        'content': json.dumps(weather)
    })

    return weather


#### MAIN APP ####
st.title('Lab 5: The "What to Wear" Bot')

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Keeps the last answer on screen after an unrelated rerun
if 'Lab5_advice' not in st.session_state:
    st.session_state.Lab5_advice = None
    st.session_state.Lab5_weather = None

location = st.text_input('Where are you headed today?',
                         placeholder=DEFAULT_LOCATION)

if st.button('What should I wear?'):
    question = location.strip() or DEFAULT_LOCATION

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'What should I wear in {question} today?'}
    ]

    client = st.session_state.openai_client

    # First call - the model decides whether it needs the weather
    with st.spinner('Checking the forecast...'):
        response = client.chat.completions.create(
            model='gpt-5-mini',
            messages=messages,
            tools=[weather_tool],
            tool_choice='auto',
        )

        reply = response.choices[0].message

        if reply.tool_calls:
            try:
                weather = run_weather_tool(reply.tool_calls[0], messages, reply)
            except Exception as e:
                st.error(f'Could not get the weather: {e}')
                st.stop()
        else:
            weather = None

    if weather:
        st.caption(f'Weather for {weather["matched_location"]}')

    # Second call - now the model can see the weather and give advice
    stream = client.chat.completions.create(
        model='gpt-5-mini',
        messages=messages,
        stream=True,
    )
    advice = st.write_stream(stream)

    if weather:
        with st.expander('The data the bot used'):
            st.json(weather)

    st.session_state.Lab5_advice = advice
    st.session_state.Lab5_weather = weather

elif st.session_state.Lab5_advice:
    weather = st.session_state.Lab5_weather
    if weather:
        st.caption(f'Weather for {weather["matched_location"]}')
    st.markdown(st.session_state.Lab5_advice)
    if weather:
        with st.expander('The data the bot used'):
            st.json(weather)
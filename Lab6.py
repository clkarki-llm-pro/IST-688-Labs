# Lab6.py
import os
import streamlit as st
import requests
from langchain.agents import create_agent

# LangChain's OpenAI integration reads the key from this environment
# variable instead of taking it as an argument like OpenAI(api_key=...)
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

DEFAULT_LOCATION = 'Syracuse, NY'

#### GET THE WEATHER ####
# location can be a city, a zip code, an airport code ('SYR'),
# or a landmark ('Eiffel+Tower')
# note: hard codes units to degrees Fahrenheit
def get_current_weather(location: str = DEFAULT_LOCATION) -> dict:
    """Get current conditions and today's hourly forecast for a location.
    Use this before giving any advice about clothing or outdoor plans.

    Returns the current temperature, feels-like temperature, conditions,
    wind, humidity, and precipitation, plus today's high and low, sunrise
    and sunset, and a forecast in three-hour steps through the day.
    Temperatures are in degrees Fahrenheit and wind speed is in miles per
    hour.

    Args:
        location: City and state or country (for example "Syracuse, NY"
            or "Lima, Peru"), a zip code, an airport code, or a landmark
            (for example "Eiffel Tower"). Omit if the user did not name a
            place; it defaults to Syracuse, NY.
    """
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


#### MAIN APP ####
st.title('Lab 6: The "What to Wear" Bot, with LangChain')

# Create the agent. create_agent replaces the tool JSON, the two API calls,
# and the message bookkeeping from Lab 5
if 'Lab6_agent' not in st.session_state:
    st.session_state.Lab6_agent = create_agent(
        model='openai:gpt-5-mini',
        tools=[get_current_weather],
        system_prompt=SYSTEM_PROMPT,
    )

# Keeps the last answer on screen after an unrelated rerun
if 'Lab6_result' not in st.session_state:
    st.session_state.Lab6_result = None

location = st.text_input('Where are you headed today?',
                         placeholder=DEFAULT_LOCATION)

if st.button('What should I wear?'):
    question = location.strip() or DEFAULT_LOCATION

    # One call runs the whole loop: the agent asks for the weather, runs
    # get_current_weather, then writes the answer
    try:
        with st.spinner('Checking the forecast...'):
            result = st.session_state.Lab6_agent.invoke(
                {'messages': [{'role': 'user',
                               'content': f'What should I wear in {question} today?'}]}
            )
    except Exception as e:
        st.error(f'Could not get your advice: {e}')
        st.stop()

    st.session_state.Lab6_result = result

if st.session_state.Lab6_result:
    # The answer is the last message in the conversation
    with st.chat_message('assistant'):
        st.write(st.session_state.Lab6_result['messages'][-1].content)

    # result['messages'] holds the whole conversation, not just the answer
    with st.expander('Behind the scenes'):
        for message in st.session_state.Lab6_result['messages']:
            st.write(f'**{message.type}**')
            st.write(message.content)
            if message.type == 'ai' and message.tool_calls:
                st.write(message.tool_calls)
# Lab6.py
#### LAB QUESTIONS (Part D, step 12) ####
# a. How many messages are there, and what is each one?
#    There are 4 messages for one city (I ran "Syracuse, NY"):
#    1. human - my question: "What should I wear in Syracuse, NY today?"
#    2. ai    - the model's request to call get_current_weather. It has no
#               text of its own
#    3. tool  - the weather data my get_current_weather function returned,
#               as a JSON string
#    4. ai    - the final answer: conditions, what to wear, outdoor plans
#    The system prompt is not in this list. create_agent keeps it separate
#    and sends it to the model with every call.
#
# b. Which message is the model asking for the weather tool? 
#    It is message 2, the first ai message. The request is not text, so
#    content is empty. It is stored in the message's tool_calls list. That
#    list has one entry with the tool name (get_current_weather), the args
#    ({'location': 'Syracuse, NY'}), an id (call_...), and the type
#    'tool_call'.
#
# c. In Lab 5, which lines of your code created each of those messages?
#    1. human: the user dict in the messages = [...] list in the button
#       block, {'role': 'user', 'content': f'What should I wear in
#       {question} today?'}. The system dict above it in that list matches
#       system_prompt=SYSTEM_PROMPT in create_agent.
#    2. ai (tool request): the first client.chat.completions.create call
#       with tools=[weather_tool] returned reply. Then run_weather_tool
#       read the arguments with json.loads(call.function.arguments) and
#       saved the request with messages.append(reply.to_dict()).
#    3. tool: the messages.append({'role': 'tool', ...}) in
#       run_weather_tool. It runs get_current_weather(location), puts
#       json.dumps(weather) in 'content', and copies the request's id into
#       'tool_call_id' so the API can match result to request.
#    4. ai (final answer): the second client.chat.completions.create call
#       with stream=True, and advice = st.write_stream(stream). Lab 5 never
#       added this answer to messages. It only displayed it and saved it in
#       st.session_state.Lab5_advice.

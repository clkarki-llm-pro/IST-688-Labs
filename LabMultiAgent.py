# LabMultiAgent.py
import asyncio
import os
import time
import streamlit as st
import requests
from agents import Agent, Runner, function_tool

# The Agents SDK reads the key from this environment variable, the same
# way LangChain did in Lab 6
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

MODEL = 'gpt-5-mini'
DEFAULT_LOCATION = 'Syracuse, NY'

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

#### THE TOOL ####
# @function_tool turns this wrapper into a tool object only an agent can
# call. get_current_weather above stays a normal function I can still call.
@function_tool
def get_weather(location: str = DEFAULT_LOCATION) -> dict:
    """Get current conditions and today's forecast for a location.

    Returns the current temperature, feels-like temperature, conditions,
    wind, humidity, and precipitation, plus today's high and low, sunrise
    and sunset, and a forecast in three-hour steps through the day.
    Temperatures are in degrees Fahrenheit and wind speed is in miles per
    hour.

    Args:
        location: City and state or country (for example "Syracuse, NY"
            or "Lima, Peru"), a zip code, an airport code, or a landmark.
            Defaults to Syracuse, NY if the user did not name a place.
    """
    return get_current_weather(location)

#### THE SPECIALISTS ####
# Each specialist only sees the text the manager sends it - not the user's
# question, and not what the other specialists said
weather_agent = Agent(
    name='Weather agent',
    instructions=(
        'You report the weather. Call get_weather for the location you are '
        'given, then summarize the facts: the place the weather is for, '
        'current conditions and feels-like temperature, today\'s high and '
        'low, how temperature, rain chance, and wind change through the '
        'day, and sunrise and sunset.\n\n'
        'Facts only - do not give clothing or activity advice.\n'
        'If the tool reports an error, say plainly that the location could '
        'not be found. Never make up weather.'
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name='Stylist agent',
    instructions=(
        'You are a stylist. Based only on the weather summary you are '
        'given, recommend what to wear today: layers, outerwear, footwear, '
        'and anything needed for rain, wind, or sun. Point out when the '
        'outfit should change through the day. Do not invent weather that '
        'is not in the summary. Keep it to a few short bullet points.'
    ),
    model=MODEL,
)

activity_agent = Agent(
    name='Activity agent',
    instructions=(
        'You plan the day. Based only on the weather summary you are given, '
        'suggest two or three outdoor activities that suit the conditions '
        'and tie each one to the time of day that suits it best. If the '
        'weather is bad all day, suggest indoor options instead. Do not '
        'invent weather that is not in the summary. Keep it brief.'
    ),
    model=MODEL,
)

#### THE MANAGER ####
# as_tool wraps a whole agent so the manager can call it like a function.
# The specialist runs, returns its answer to the manager, and the manager
# stays in charge of the final reply.
manager_agent = Agent(
    name='Manager',
    instructions=(
        'You coordinate a team that answers "what should I wear and do '
        'today?" for a location.\n\n'
        '1. Always call weather_report first, with the location.\n'
        '2. If the weather report says the location could not be found, '
        'do not call the other tools. Tell the user you could not find '
        'that place and suggest a city with its state or country.\n'
        '3. Otherwise, send the full weather summary to style_advice and '
        'to activity_ideas. They cannot see the weather report or each '
        'other, so include everything they need in what you send.\n'
        '4. Combine their answers into one self-contained reply: start by '
        'naming the place the weather is for, then a short line on '
        'conditions, then clothing, then two or three activities.\n\n'
        'Do not mention the team, the tools, or the lookup. Do not ask the '
        'user follow-up questions. They cannot reply.'
    ),
    tools=[
        weather_agent.as_tool(
            tool_name='weather_report',
            tool_description='Looks up today\'s weather for a location and '
                             'returns a factual summary. Send it the '
                             'location.'),
        stylist_agent.as_tool(
            tool_name='style_advice',
            tool_description='Recommends what to wear for the day. Send it '
                             'the full weather summary.'),
        activity_agent.as_tool(
            tool_name='activity_ideas',
            tool_description='Suggests activities and the best time of day '
                             'for each. Send it the full weather summary.'),
    ],
    model=MODEL,
)

#### THE SINGLE AGENT (Part E) ####
# Same job as the whole team, done by one agent with one prompt. This is
# my Lab 6 system prompt with the tool name changed to get_weather.
SYSTEM_PROMPT = (
    'You are a "what to wear" assistant. Given a location, look up the '
    'weather and then tell the user what to wear today and suggest outdoor '
    'activities that suit the conditions.\n\n'
    'Do not guess at the weather - always use the get_weather tool.\n\n'
    'Write the answer as a single self-contained response:\n'
    '- Do not mention the lookup, the tool, or the data you were given.\n'
    '- Note how conditions change through the day when it affects the '
    'advice, and tie activity suggestions to the times that suit them.\n'
    '- Do not ask the user follow-up questions. They cannot reply.\n'
    '- Keep it brief - a short line on conditions, then clothing, then '
    'two or three activities.'
)

single_agent = Agent(
    name='Single agent',
    instructions=SYSTEM_PROMPT,
    tools=[get_weather],
    model=MODEL,
)


#### RUN AN AGENT ####
# Runner.run_sync fails on the second click with "Event loop is closed":
# each Streamlit rerun gets a new event loop, but the SDK's OpenAI client
# stays tied to the first one. Caching one loop for the app avoids that.
@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()

def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


#### MAIN APP ####
st.title('Lab 9: The "What to Wear" Crew')

location = st.text_input('Where are you headed today?',
                         placeholder=DEFAULT_LOCATION)

mode = st.radio('Who answers?', ['Team', 'Single agent'], horizontal=True)
agent = manager_agent if mode == 'Team' else single_agent

if st.button('Get advice'):
    question = location.strip() or DEFAULT_LOCATION

    try:
        with st.spinner('Checking the weather and planning your day. '
                        'This can take up to a minute...'):
            result, seconds = run(agent,
                                  f'What should I wear and do today in {question}?')
    except Exception as e:
        st.error(f'Could not get your advice: {e}')
        st.stop()

    with st.chat_message('assistant'):
        st.write(result.final_output)

    # usage adds up every model request in the run; new_items is the
    # manager's step-by-step record of what happened
    usage = result.context_wrapper.usage
    with st.expander('Behind the scenes'):
        st.write(f'Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · '
                 f'Time: {seconds:.1f}s · Answered by: {result.last_agent.name}')
        for item in result.new_items:
            if item.type == 'tool_call_item':
                st.write(f'{item.agent.name} called {item.raw_item.name}: '
                         f'{item.raw_item.arguments}')
            elif item.type == 'tool_call_output_item':
                st.write(f'result: {str(item.output)[:300]}')
            elif item.type == 'handoff_output_item':
                st.write(f'handoff: {item.source_agent.name} -> {item.target_agent.name}')

#### LAB QUESTIONS (Part D) ####
# I ran "Syracuse, NY"
# Model requests: 7 | Tokens: 7,516 | Time: 53.2s
# It was answered by Manager
#
# a. Which tools did the manager call, and in what order?
# First it called weather_report with "Syracuse, NY". Then it called
# style_advice and activity_ideas. get_weather does not show up because
# the weather agent called it and not the manager.
#
# b. What did the manager send to the stylist: the full forecast from the
# tool, or a summary? Who wrote that text?
# It sent a summary, not the full forecast. The weather agent wrote the
# summary from the tool's data, and the manager passed it on to the
# stylist. So a model wrote that text.
#
# c. "Model requests" is higher than the number of tool calls you see.
# Where did the other model calls happen?
# I only see 3 tool calls, but there were 7 model requests. The extra
# calls happened inside the specialist agents, because each one is its
# own agent that makes its own model calls, and the manager also makes
# a call to write the final answer.
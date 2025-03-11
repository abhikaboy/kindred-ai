# Please install OpenAI SDK first: `pip3 install openai`

import json
from openai import OpenAI
import os

api_key = os.environ.get("deepseek_key")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

schema = str(open("schema.schema.json").read())
system_prompt = """
The user will provide a natural language description of a task. You will extract the following information from the user's description:

- The task's priority (1 - high, 2 - medium, 3 - low)
- The task's content (a concise description of the task)
- The task's difficulty (1 - very easy, 2 - easy, 3 - medium, 4 - hard, 5 - extremely hard)
- Whether the task is recurring (true or false)
- The task's recurrence details (if the task is recurring, provide the frequency, days of the week, and the next date)
- Weekly tasks will be a list of days of the week (true or false) starting on MONDAY
- Whether the task is public (true or false)
- Whether the task is active (true or ifalse)
- The task's deadline (if the task has a deadline, provide the date)
- The task's start day (if the task has a start day, provide the date)
- The task's start time (if the task has a start time, provide the time)

Please use the following JSON Schema: 
  {schema}

  The priority and difficulty also wont always be specified, and will sometimes need to be inferred. 
  If you could infer the priority and difficulty, notes, and a basic checklist please do so.
  The checklist can be between 1 to 10 items.
  Please refer all dates in reference to the current date. (March 10 2025)
  Please pay attention to dates and times the user could refernce.

  Here are some example outputs: 
  "output": { 
      "priority": 2, 
      "content": "Meditate for 15 minutes", 
      "difficulty": 1, 
      "recurring": true, 
      "recurDetails": { 
          "frequency": "DAY", 
          "every": 1, 
          "nextDate": "2025-03-10" 
      }, 
      "public": true, 
      "active": true,
      "startTime": "19:00",
      "deadline": "2025-03-10T23:59"
  }

  Please pay EXTRA importance to the schema and ensure it is matched exactly - especially the recurrence details.
  
  Weekly recurrences should be in the following format:
  "recurDetails": { 
      "frequency": "WEEK", 
      "days": [false, false, false, false, false, false, true], 
      "nextDate": "YYYY-MM-DD"
  }

  If necessary you can seperate the task into multiple tasks if you think that would make things better!
  For every request, please specify up to 2 different options.
   """


goal_prompt = """

The user will provide a goal or habit that they want to achieve. Your job is to identify 
an appropriate amount of tasks that will help them achieve their goal. Please 
respond with a series of sentences, not tasks that will help them achieve their goal.

The response format should be a JSON object with an array (called tasks) of object with the following keys:

- rank (a number between 1 and 10)
- sentence (a sentence that will help them achieve their goal)
- recurring (true or false)

Please limit the number of options to 5.

"""

def create_task(task=None):
  if task is None:
    task = input("Please provide a task: ")
  

  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": task},
      ],
      stream=False,
      response_format={
          'type': 'json_object'
      })
  return response.choices[0].message.content

def create_goal():
  goal = input("Please provide a goal: ")

  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=[
          {"role": "system", "content": goal_prompt},
          {"role": "user", "content": goal},
      ],
      response_format={
          'type': 'json_object'
      },
      stream=False)
  # ret = []
  # for choice in response.choices:
  #   ret.append(str(choice.message.content))
  # return ret
  return response.choices[0].message.content

goals = json.loads(create_goal())
print(goals.keys())
for task in goals["tasks"]:
  print(task.keys())
  print(create_task(task["sentence"]))
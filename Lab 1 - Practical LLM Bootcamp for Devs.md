---
title: Lab 1 - Practical LLM Bootcamp for Devs

---

# Practical LLM Bootcamp for Devs

## Week 1 - Lab: Evaluation

Evaluation is a critical component of successfully building LLM solutions. You can develop an 80% working prototype very quickly, but bringing your solution to production standards is incredibly difficult.

There are many evaluation libraries and providers, here are some top choices:

- LangSmith. Made by the creators of LangChain, an early mover in the space and popular. Some think their framework is overly convoluted.
- Langfuse. An open source alternative to LangSmith.
- DeepEval. A lighter weight open source solution.
- MLFlow. Origins in machine learning, but has expanded to encompass LLM.

In this lab, we're going to explore LangSmith, but most of them share similar concepts. A few common use cases:

- Trace calls to the LLM
- Create datasets to be used for evaluation or fine-tuning
- Use a variety of evaluators to score the LLM responses. For many apps, the best evaluation is LLM-as-a-judge.

### Milestone 1 - Exploring LangSmith basics

Databricks [released an LLM](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), and let's say we wanted to build a chat bot to answer questions about it.

1. Create a new folder, and create a virtual environment, and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -U langsmith openai
```

3. Create .env

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_API_KEY
LANGCHAIN_PROJECT="LangSmith Demo"
```

4. Create `create_dataset.py`

```python
from langsmith import Client

client = Client()
dataset_name = "DBRX"

inputs = [
    "How many tokens was DBRX pre-trained on?",
    "Is DBRX a MOE model and how many parameters does it have?",
    "How many GPUs was DBRX trained on and what was the connectivity between GPUs?",
]

outputs = [
    "DBRX was pre-trained on 12 trillion tokens of text and code data.",
    "Yes, DBRX is a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters.",
    "DBRX was trained on 3072 NVIDIA H100s connected by 3.2Tbps Infiniband",
]

# Store
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA pairs about DBRX model.",
)
client.create_examples(
    inputs=[{"question": q} for q in inputs],
    outputs=[{"answer": a} for a in outputs],
    dataset_id=dataset.id,
)
```

Run create_dataset.py, then go to the LangSmith dashboard and view your dataset.

5. Create `eval.py`

```python
import openai
from langsmith.wrappers import wrap_openai

import requests
from bs4 import BeautifulSoup

url = "https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = [p.text for p in soup.find_all("p")]
full_text = "\n".join(text)

openai_client = wrap_openai(openai.Client())

def answer_dbrx_question_oai(inputs: dict) -> dict:
    """
    Generates answers to user questions based on a provided website text using OpenAI API.

    Parameters:
    inputs (dict): A dictionary with a single key 'question', representing the user's question as a string.

    Returns:
    dict: A dictionary with a single key 'output', containing the generated answer as a string.
    """

    # System prompt
    system_msg = (
        f"Answer user questions in 2-3 sentences about this context: \n\n\n {full_text}"
    )

    # Pass in website text
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]

    # Call OpenAI
    response = openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )

    # Response in output dict
    return {"answer": response.dict()["choices"][0]["message"]["content"]}
  
  
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Evaluators
qa_evalulator = [LangChainStringEvaluator("cot_qa")]
dataset_name = "DBRX"

experiment_results = evaluate(
    answer_dbrx_question_oai,
    data=dataset_name,
    evaluators=qa_evalulator,
    experiment_prefix="test-dbrx-qa-oai",
    # Any experiment metadata can be specified here
    metadata={
        "variant": "stuff website context into gpt-3.5-turbo",
    },
)
```

Go to your dataset, and go to the experiments tab to see the results of your evaluation.

### Milestone 2 - Add tracing

1. Clone https://github.com/timothy1ee/llm_tutor
Follow the installation instructions, and add your OpenAI key to .env (Runpod not required)
   - Experiment interacting with the tutor, try changing the system prompt
2. Create a LangSmith account
   - Follow steps 1-4 here: https://docs.smith.langchain.com/
   - Add the @traceable annotation to all functions in the llm_tutor

When you finish this milestone, you should be able to inspect each function call you added @traceable to.
 
### Milestone 3: Create a dataset

1. Interact with the prompt, then go to the LangSmith web ui and pick 5 representative examples. A representative example is one that represents a typical user interaction with your app, with an ideal response from the LLM.
2. Add the chosen examples to a new dataset

### Milestone 4: Add a custom LLM-as-a-judge evaluation

1. Create a new file, `eval.py`

```
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Add print statements to explore inputs and outputs

  return {
    "key": "prompt_compliance",
    "score": 0,  # Normalize to 0-1 range
    "reason": "None"
  }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "Python tutoring"

# A string to prefix the experiment name with.
experiment_prefix = "Python tutoring prompt compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)
```

At this point, you're just demonstrating that you can pull down the dataset. Also, you need to explore the structure of the items in the dataset.

The relevant data is in the `example` parameter. Run `python eval.py` and print out `example.inputs` and `example.outputs` and note what they are.

Now, we want to pass all the context to the judge, and have it score how well the model is adhering to the prompt.

```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']

    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. 
    Provide a score from 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.
    Also provide a brief explanation for your score.

    Respond in the following JSON format:
    {{
        "score": <int>,
        "explanation": "<string>"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "score": result["score"] / 10,  # Normalize to 0-1 range
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
        return {
            "key": "prompt_compliance",
            "score": 0,
            "reason": "Failed to parse evaluator response"
        }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "Python tutoring"

# A string to prefix the experiment name with.
experiment_prefix = "Python tutoring prompt compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)
```

### Milestone 5: Improve the judge

Read this section on optimizing an LLM judge: https://huggingface.co/learn/cookbook/en/llm_judge#3-improve-the-llm-judge

Make improvements on the prompt, and see if you can increase the relative score.

### Milestone 6: Create a dataset for the student assessment prompt

Create a dataset for the model that is assessing the student's knowledge and monitoring for alerts.

### Milestone 7: Add a custom LLM-as-a-judge evaluation

Add another evaluation script to run an experiment on the student assessments.

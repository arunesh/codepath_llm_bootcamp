---
title: AI/LLM Bootcamp - Prework

---

# AI/LLM Bootcamp - Prework

Over the 6 weeks of the bootcamp, you will become comfortable architecting LLM solutions for a wide range of problems. Those solutions will leverage more sophisticated prompt engineering, RAG, fine-tuning, tools/agents, evaluation, and the orchestration of ensemble solutions.

In this prework, we'll help you hit the ground running by building a chat interface, similar to ChatGPT, and hosting your own LLM model. By doing this, we'll get some basic accounts set up (OpenAI, HuggingFace, RunPod), we'll make sure your Python environment is configured, and we'll gain some insight on the basic building blocks of building an LLM app.

In the video below, I'll demo what you're going to build, and I'll walk you through a series of small milestones to build a simple LLM application.

**Watch the video below:**

{%youtube VXVSu6cV-sQ %}

## Milestone 1: Chainlit and Python

This bootcamp is primarily about LLM backend architecture. That said, it's convenient to have various frontend solutions to interact with and evaluate your LLM solution.

In this milestone, you'll set up Chainlit, an open source Python framework which provides a ChatGPT-like GUI interface. By the end of this milestone, you'll also have your Python environment set up. Note: there are several popular, open source alternatives that do similar things. Chainlit is easy to set up and works well for simple use cases.

### Step 1: Install Python

* If you're using OS X, I recommend that you install Python 3 with Homebrew. If you don't already use Homebrew, [install here](https://brew.sh/).
  * In Terminal, run `brew install python`
* If you're using Windows, I recommend that you install via [direct download](https://www.python.org/downloads/).

Note: if you choose to download Python directly on OS X, that should work pretty well too.

### Step 2: Create the Chainlit app

1. Create the project folder

```
mkdir chainlit_demo
cd chainlit_demo
```

2. Create the virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

- Note: if you're new to Python development, read more about virtual environments [here](https://hackmd.io/ilnbFySpR6Ok_59U_74hFg?view).

3. Install chainlit

```
pip install chainlit
pip freeze > requirements.txt
```

4. Add .gitignore

- Add this [Python .gitignore template](https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore)

5. Create app.py

- Use your favorite Python IDE, and if you don't have one, use [VS Code](https://code.visualstudio.com/), it's great.
- Create a new file called app.py, and copy the file below

```
import chainlit as cl

@cl.on_message
async def on_message(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()
```

6. Run the chainlit app

```
chainlit run app.py -w
```

Note: the `-w` flag watches for changes to your app.py and automatically restarts the server to apply any code changes.

7. Push to GitHub

- Run `git init`, add/commit your files, then push to a new GitHub repository.
- Note: if you're new to Python, when you clone a Python project for the first time, you have to create the virtual environment, and install the packages in `requirements.txt`. You don't have to do this here because you set up the virtual environment already. This is just if you're setting up your project from another computer.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

At this point, you've completed Milestone 1, and you have a basic Chainlit app, and your Python environment is set up.

## Milestone 2: Integrating OpenAI

Now that we have your frontend, we want to wire it up with an actual LLM, so you can chat with it. In order to do that, we'll create an OpenAI platform account, which is different from a ChatGPT Pro account.

### Step 1: Setup your OpenAI platform account

1. Create an OpenAI platform account: https://platform.openai.com/
2. Go to billing, and add $10
3. Create a project, and a secret key: https://platform.openai.com/api-keys

![Screenshot 2024-08-17 at 11.18.46 AM](https://hackmd.io/_uploads/SkMpKDRc0.png =50%x)

![Screenshot 2024-08-17 at 11.24.38 AM](https://hackmd.io/_uploads/S1sUoP05A.png)

### Step 2: Setup your .env file

You don't want to embed your api keys in your project files on GitHub. Instead, store them in an .env file (which is already being ignored in .gitignore).

1. Add a .env file
2. In the .env file, add the following line. Don't put quotes around your api key

```
OPENAI_API_KEY=sk-proj-QLg3...
```
3. Create .env.sample, which you'll check in, so you'll remember which keys you're using. Your .env.sample can look like the following.

```
OPENAI_API_KEY=YOUR_KEY_HERE
```

### Step 3: Install OpenAI

Make sure you are using a Terminal where you have activated your virtual environment and install the Python OpenAI library.

Every time you install a new dependency, remember to update your requirements.txt file.

```
pip install openai
pip freeze > requirements.txt
```

### Step 4: Modify app.py

Modify your app.py file to fetch the api key, create an OpenAI API client, then send the message to OpenAI in the message handler.

If you're new to asynchronous programming in Python, read up on using async/await. When you put `await` in front of a long running operation like a network call, it allows you to pretend that it's executing synchronously, but without blocking the interrupt handler.

```
import chainlit as cl
import openai
import os

api_key = os.getenv("OPENAI_API_KEY")

endpoint_url = "https://api.openai.com/v1"
client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.3,
    "max_tokens": 500
}

@cl.on_message
async def on_message(message: cl.Message):
    # Your custom logic goes here...

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": message.content}],
        **model_kwargs
    )

    # https://platform.openai.com/docs/guides/chat-completions/response-format
    response_content = response.choices[0].message.content

    await cl.Message(
        content=response_content,
    ).send()
```

You've completed Milestone 2, and you have a bot that you can chat with! This is like ChatGPT, but with the ability to modify temperature.

Try a prompt like: "Imagine an LLM has unlimited creativity. Generate five futuristic and groundbreaking LLM business ideas that seem almost impossible today but could be revolutionary in the future." Try the prompt with a temperature of 0.2 vs 1.2. You can also increase the number of max tokens.

Higher temperatures can achieve higher levels of creativity, but can be less rational and have more hallucination.

## Milestone 3: Improving the Chat experience

LLMs are stateless. Currently, your app sends the latest message only, and gets a reply. In order to maintain a conversation, ChatGPT manually keeps track of the conversation history, and passes in the entire conversation history with each request.

LangChain does something similar. As conversations get too long to fit into the context, LangChain will attempt to first summarize the conversation to compact what is sent to the model.

Also, text generation can be slow. For a better user experience, we can stream the characters as they become available.

### Step 1: Convert to streaming

In app.py, modify the on_message handler to stream tokens to the response_message as they become available.

```
@cl.on_message
async def on_message(message: cl.Message):
    response_message = cl.Message(content="")
    await response_message.send()
    
    stream = await client.chat.completions.create(messages=[{"role": "user", "content": message.content}], 
                                                  stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
```

### Step 2: Add conversation history

Leverage a feature of Chainlit, which allows you to persist data across requests in a user_session dictionary.

```
@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()
    
    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history, 
                                                  stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
```    

You've completed Milestone 3, and your chat bot now feels more responsive, and remembers the entire conversation.

## Milestone 4: Running your own LLM model

There are dozens of popular open source models that research teams have developed for different purposes. Your final AI/LLM architecture may likely leverage several LLM models that work in concert with each other. You may also fine-tune a model to improve its performance for a specific set of use cases.

In order to do this, you need to be able to host an LLM model for inference. Inference is the process of generating output.

In this model, we will host a Mistral model, which is one of the most popular high performing models. You can use this same technique to host Llama, Qwen, Yi, etc.

**Watch the video below:**

{%youtube BvAFsbCL5HA %}

### Step 1: Setup accounts

1. Create a Runpod account: https://runpod.io?ref=qofw05lr
  - Add $10. https://www.runpod.io/console/user/billing
  - Create an API key: https://www.runpod.io/console/user/settings
2. Create a Hugging Face account
    - https://huggingface.co/join
    - Create an access token: https://huggingface.co/settings/tokens
    - You only need the "Read" permission to download the open source models

### Step 2: Accept model terms on HuggingFace

Many models are "gated" models. In order to download them, you must first login to HuggingFace, go to the model page, and review and accept their terms.

Once you do that, if you provide your Hugging Face token in an API call, it'll allow you to download the model.

Go here to accept the terms for the Mistral 7B Instruct 0.3 model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

![Screenshot 2024-08-17 at 9.48.23 PM](https://hackmd.io/_uploads/B1_8Tl1iC.png)

### Step 3: Configure RunPod

Follow the steps below to configure your RunPod serverless instance. There are many cloud GPU and LLM providers. RunPod is one of the cheapest, and it's also the only serverless provider that I know of that allows you to run your own fine tuned model.

Other serverless solutions either only allow you to run an existing common model, or require you to keep the endpoint running all the time (> $1k / mo). RunPod allows you to have a serverless endpoint that goes to sleep. The downside is that it can take a minute to wake up after it's idle, but it's a great prototyping solution.

1. Create a network volume

  - Go here: https://www.runpod.io/console/user/storage
  - Give it any label (e.g., Mistral 7B Instruct 0.3)
  - Choose a data center with good availability of RTX A6000
  - For Mistral, give it 30GB of storage.
  - Note: as long as you have storage, it will charge you. Delete the storage to stop the charges. It's just for caching the model weights for faster startup.

2. Create a serverless endpoint

  - Go here: https://www.runpod.io/console/serverless
  - Endpoint name: Mistral 7B Instruct 0.3 (label doesn't really matter)
  - 48 GB GPU
  - Consider setting idle timeout to 600 seconds during testing and development (may increase cost)
  - Container image: runpod/worker-v1-vllm:stable-cuda12.1.0
  - Environment variables
    - HF_HUB_ENABLE_HF_TRANSFER => 1
    - HF_TOKEN => hf_sdNY... (replace with your Hugging Face token)
    - MODEL_NAME => mistralai/Mistral-7B-Instruct-v0.3
  - Expand the Advanced options
    - Select Network volume: choose the network volume that you created earlier
    - Allowed CUDA versions: 12.1, 12.2, 12.3
  - Reference: https://github.com/runpod-workers/worker-vllm. You don't need to view that repo, but just in case you're curious about how we knew which environment variables to set, and what is contained in the Docker image.

### Step 4: Modify app.py

1. Add `RUNPOD_API_KEY` to your .env file.
2. Add `RUNPOD_SERVERLESS_ID` to your .env file. Find the id in your serverless endpoint page (see image below)

![Screenshot 2024-08-17 at 10.44.12 PM](https://hackmd.io/_uploads/rknpcZyoA.png =50%x)

3. Modify app.py as below to switch to using your RunPod key and url.

Note that we'll still use the OpenAI client. Many LLM API providers will use the same API schema as OpenAI, which makes it easier to switch between LLM models.

```
# api_key = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("RUNPOD_API_KEY")
runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")

# endpoint_url = "https://api.openai.com/v1"
endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"

client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
# model_kwargs = {
#     "model": "chatgpt-4o-latest",
#     "temperature": 1.2,
#     "max_tokens": 500
# }

model_kwargs = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "temperature": 0.3,
    "max_tokens": 500
}
```

You've completed Milestone 4, and you should be able to interact with your Mistral LLM model.

## Milestone 5: Supporting image attachments

One of the exciting developments is the capabilities of models to understand images, including screenshots of complex PDFs.

Mistral doesn't support images, so you'll need to switch back to gpt4o. Read more about [gpt4o vision here](https://platform.openai.com/docs/guides/vision).

Copy the code below into your `app.py` file. For a real app, it's better to upload your images to a bucket somewhere, and pass the url to gpt4o. This allows them to cache the image.

If we pass as base64 data, we have to pass the image with each chat message, since we send the entire chat conversation with each message.

```
import base64

@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    # Processing images exclusively
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read the first image and encode it to base64
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content if message.content else "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()
    
    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history, 
                                                  stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
```

You've completed Milestone 5, and you can try passing in various images, including images of a PDF. You'll find it does a pretty impressive job of understanding what's on the page.

## Milestone 6: Submission

After pushing your code to GitHub, submit your repository by filling out the [form here](https://forms.gle/LQqsbVoQaokPi8JU6). Make sure the repo is publicly visible.

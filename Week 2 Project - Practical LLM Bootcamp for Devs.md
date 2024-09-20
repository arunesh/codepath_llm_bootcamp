---
title: Week 2 Project - Practical LLM Bootcamp for Devs

---

# Practical LLM Bootcamp for Devs

## Week 2 - Project: Retrieval Augmented Generation

Last week, we discussed the general architecture of LLM solutions, the mechanics of evaluation, and some underlying theory of neural networks.

Here are some discussion questions from last week as a recap:

- Can LLMs access the internet? If not, how does ChatGPT do it?
- How do LLMs remember your conversation?
- An LLM is a neural network. Some questions about neural networks:
  - What is a parameter in a neural network?
  - During the training or fine-tuning process, when you run a piece of training data through the LLM, what happens to the parameters?
  - What is an epoch?
  - What's the difference between a base model, a chat model, and a coding model?

This week, we will explore an important concept in designing LLM solutions, which is the ability to fetch appropriate data and add that context to the prompt. This technique is called Retrieval Augmented Generation.

### Milestone 1: RAG with PDFs

We're going to use LlamaIndex to allow us to ask questions of a PDF. We'll also explore using Jupyter Notebooks for a more interactive environment.

1. Create a new Python project (make a directory and create a virtual environment)

```bash!
mkdir rag_demo
cd rag_demo

python3 -m venv .venv
source .venv/bin/activate
```
2. Create a subfolder called `data`
    - Add the [pdf of CodePath's strategic plan](https://drive.google.com/file/d/1eHaP8M0UbJ7ho06m-NyGb31vtuxDJahb/view) to the data subfolder.
    - Note: you can add any number of documents in the `data` folder. See the [supported file types here](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/).
3. Install dependencies
```bash!
pip install python-dotenv llama-index
```
4. Create a .env file
   - Add OPENAI_API_KEY=sk-proj-... to your .env file
5. Create a Jupyter Notebook file called `rag_demo.ipynb`
```python!
# Import necessary libraries
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Example query
response = query_engine.query("What years does the strategic plan cover?")

print(response)
```
7. Choose the Python (.venv) kernel. This ensures that the Jupyter notebook runs its code in the virtual environment that you created. "Select kernel" -> ".venv"
![Screenshot 2024-09-16 at 9.31.17 AM](https://hackmd.io/_uploads/B15s6CHTC.png =75%x)
8. Run the code cell above, and see the answer.

Note: you can create an additional cell and ask more questions. This is convenient because you don't need to re-run the previous cells, everything is still in memory.

### Milestone 2: Gmail access (optional)

The following instructions will configure your Google Workspace or personal Gmail account for access to the Gmail API. You'll get a file called "credentials.json", which you can use later to get an access token.

1. Create a [project in Google Cloud](https://console.cloud.google.com/projectcreate)
    - I called my project, "RAG email"
2. Go to APIs & Services -> Library, search for Gmail, and enable it
3. Go to APIs & Services -> Oauth consent screen and create a consent screen
    - Use internal if you're using a Google Workspace account
    - Use external if you're using a Gmail account
    - Fill out required fields
    - Add scope, search for gmail, and choose "gmail.readonly"
    - If you had to create an external app, add yourself as a test user
4. Go to APIs & Services -> Credentials and create a new OAuth client ID
    - Application type: Desktop
    - ![Screenshot 2024-09-16 at 5.53.52 AM](https://hackmd.io/_uploads/H1I0gCr6R.png =75%x)
5. Download the json file for the client id, and save in your project file as "credentials.json"
![Screenshot 2024-09-16 at 6.07.09 AM](https://hackmd.io/_uploads/Hyn8x0HpR.png =50%x)

Copy credentials.json to the new project folder, and open in Cursor or VS Code.

### Milestone 3: Download emails (optional)

Follow the instructions below to create a wrapper around the Gmail API, and create LlamaIndex Documents.

A Document is primarily a chunk of text (optionally with metadata). The chunk of text can be indexed by LlamaIndex (split into even small text blocks and indexed with a traditional search mechanism or with an embeddings model).

1. Active the Python environment and install requirements
```
pip install python-dotenv
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install llama-index
```
2. Download the [CustomGmailReader](https://gist.github.com/timothy1ee/917862e3c39481e1436b3ec5ad0267fd) to your project folder.
3. Create `rag_email.ipynb`

```python
from dotenv import load_dotenv
load_dotenv()

from custom_gmail_reader import CustomGmailReader

# Instantiate the CustomGmailReader
loader = CustomGmailReader(
    query="",
    max_results=50,
    results_per_page=10,
    service=None
)

# Load the emails
documents = loader.load_data()

# Print email information
print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents[:20]):
    print(f"Document {i+1}:")
    print(f"To: {doc.metadata.get('to', 'N/A')}")
    print(f"From: {doc.metadata.get('from', 'N/A')}")
    print(f"Subject: {doc.metadata.get('subject', 'N/A')}")
    print(f"Date: {doc.metadata.get('date', 'N/A')}")
    print(f"Content snippet: {doc.text[:1000]}...")
    print("=" * 50)
```

In the console output, you should see your last 20 emails. Instantiating the CustomGmailReader will search for a "credentials.json" file, and generate a "token.json" file. After some time, the token will expire. Something isn't working with the renewal logic -- if you get an auth error, delete the "token.json" file and allow it to be regenerated.

### Milestone 4: LlamaIndex indexing and retrieval

Similar to how we indexed the PDF, now that we have the email as Documents, we can also index and query the emails.

In `rag_email.ipynb`, add a new code block below.

```python!
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create retriever
retriever = VectorIndexRetriever(index=index)

# Create query engine
query_engine = RetrieverQueryEngine(retriever=retriever)

# Example query
response = query_engine.query("<Insert a question that can be answered from your email>")
print(response)
```

Look through your email, and test the ability of this simple RAG pipeline to answer questions.

Now that we've added inputs for PDFs and email, you can imagine extending ingestors for Slack, MMS, Notion, Asana, RSS feeds, etc.

### Milestone 5: Add tracing

In order to get a better understanding of what's happening under the hood with LlamaIndex, let's add tracing.

You can use any tracing platform with LlamaIndex, but they have streamlined the integration with [several here](https://docs.llamaindex.ai/en/stable/module_guides/observability/).

Tracing and evaluation seems like a commodity space right now, and tough for a startup to survive long term. Because of that, I like the open source options:

- Arize phoenix
- Langfuse
- Deepeval

Let's explore Langfuse, which is fairly similar to Langsmith.

1. Create a [Langfuse account](https://langfuse.com/)
2. Create a project, then create API keys (in settings)
3. Integrate your project with [these instructions](https://langfuse.com/docs/integrations/llama-index/get-started)

After setting up Langfuse, run queries to your PDF or email, and look at what's being sent to the LLM.

### Milestone 6: Exploring embeddings

A bunch of small milestones to build intuition around embeddings and what words/phrases have similarities with each other.

1. Make sure you're still in your activated virtual environment and install dependencies
`pip install scikit-learn`
2. Create `embeddings_demo.ipynb` and choose the .venv kernel

Let's add a variety of cells, and see if we can build an intuition for embeddings.

```python!
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the trivia questions and matching answers
phrases = [
    "Who was the first president of the United States?",
    "What is the capital city of France?",
    "In what year did humans first land on the moon?",
    "Which element on the periodic table has the chemical symbol O?",
    "What is the largest planet in the solar system?",
    "The first president of the United States was George Washington.",
    "The capital city of France is Paris.",
    "Humans first landed on the moon in the year 1969.",
    "The chemical symbol O represents the element Oxygen.",
    "The largest planet in the solar system is Jupiter."
]

# Generate embeddings for each phrase using OpenAI embeddings
embeddings = embedding_model.get_text_embedding_batch(phrases)

# Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Print the first phrase and the first several elements of its embedding
print(f"Phrase: {phrases[0]}")
print(f"First 5 elements of its embedding: {embeddings_array[0][:5]}\n")

# Compute cosine similarity between the embeddings
similarity_matrix = cosine_similarity(embeddings_array)

# Print the cosine similarity matrix
print("Cosine Similarity Matrix:")
print(np.round(similarity_matrix, 2))
print("\nDetailed Similarity Results:\n")

# Output comparison between phrases with improved readability
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        print(f"Cosine similarity between:\n  '{phrases[i]}'\n  and\n  '{phrases[j]}'\n  => {similarity_matrix[i, j]:.4f}\n")
```

Look at the similarity scores above, notice the typical similarity score, and notice the score for a matching question and answer pair.

Let's see how well it does if the questions and answers are all about the same topic, in this case, Astronomy.

```python!
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the new space-related questions and answers
phrases = [
    "What year did the first human land on the moon?",
    "Which planet is known as the Red Planet?",
    "What is the largest moon of Saturn?",
    "Who was the first person to travel into space?",
    "What is the name of NASA's rover that landed on Mars in 2021?",
    "The first human landed on the moon in 1969.",
    "The planet known as the Red Planet is Mars.",
    "The largest moon of Saturn is Titan.",
    "Yuri Gagarin was the first person to travel into space.",
    "NASA's rover that landed on Mars in 2021 is named Perseverance."
]

# Generate embeddings for each phrase using OpenAI embeddings
embeddings = embedding_model.get_text_embedding_batch(phrases)

# Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Print the first phrase and the first several elements of its embedding
print(f"Phrase: {phrases[0]}")
print(f"First 5 elements of its embedding: {embeddings_array[0][:5]}\n")

# Compute cosine similarity between the embeddings
similarity_matrix = cosine_similarity(embeddings_array)

# Print the cosine similarity matrix
print("Cosine Similarity Matrix:")
print(np.round(similarity_matrix, 2))
print("\nDetailed Similarity Results:\n")

# Output comparison between phrases with improved readability
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        print(f"Cosine similarity between:\n  '{phrases[i]}'\n  and\n  '{phrases[j]}'\n  => {similarity_matrix[i, j]:.4f}\n")
```

Was the score able to find the matching questions and answers?

Let's do one last one, the most challenging of all, where we have one question, one correct answer, and four wrong answers.

```python!
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the question and answers (1 correct, 4 closely related wrong ones)
phrases = [
    "What spacecraft was used in the mission to carry the first humans to the moon?",  # Question
    "Apollo 11 was the spacecraft used to carry the first humans to the moon.",       # Correct Answer
    "Apollo 12 was the spacecraft used to carry the first humans to the moon.",         # Wrong Answer
    "Apollo 14 was the spacecraft used to carry astronauts on the third successful moon landing mission.", # Wrong Answer
    "Apollo 10 was the spacecraft used to carry the first humans to the moon.", # Wrong Answer
    "Apollo 16 was the spacecraft that carried astronauts to explore the lunar highlands."   # Wrong Answer
]

# Generate embeddings for the question and answers using OpenAI embeddings
embeddings = embedding_model.get_text_embedding_batch(phrases)

# Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Print the first phrase and the first several elements of its embedding
print(f"Phrase: {phrases[0]}")
print(f"First 5 elements of its embedding: {embeddings_array[0][:5]}\n")

# Compute cosine similarity between the embeddings
similarity_matrix = cosine_similarity(embeddings_array)

print("\nDetailed Similarity Results:\n")

# Output comparison between question and answers with improved readability
for i in range(1, len(phrases)):
    print(f"Cosine similarity between the question and:\n  '{phrases[i]}'\n  => {similarity_matrix[0, i]:.4f}\n")
```

Even though Ada (the embedding model by OpenAI) is one of the most commonly used, there are many embedding models, and many are ranked higher than Ada. See the list here: https://huggingface.co/spaces/mteb/leaderboard

For specific domains, to further improve the search effectiveness, you can also fine-tune your embedding model.

In order to create the embedding models, you might use a training dataset like this: https://huggingface.co/datasets/embedding-data/QQP_triplets. This type of training set helps the model understand sentences that are semantically related and sentences that are not.

### Milestone 7: Designing evaluations

You can use an LLM to generate an evaluation dataset for your RAG pipeline.

Go through the document, chunk by chunk. Pass each chunk into an LLM, and instruct it to generate question and answer pairs. Later, you'll test if your RAG solution will be able to generate an equivalent answer.

1. Create `generate_dataset.py`

```python!
# Import necessary libraries
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

from openai import OpenAI
import json

client = OpenAI()

# Function to generate questions and answers
def generate_qa(prompt, text, temperature=0.2):    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}],
        temperature=temperature,
    )
    
    print(response.choices[0].message.content)

    # Strip extraneous symbols from the response content
    content = response.choices[0].message.content.strip()
    
    # Remove potential JSON code block markers
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
    if content.endswith('```'):
        content = content.rsplit('\n', 1)[0]
    content = content.strip()
    
    # Attempt to parse the cleaned content as JSON
    try:
        parsed_content = json.loads(content.strip())
        return parsed_content
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON. Raw content:")
        print(content)
        return []

factual_prompt = """
You are an expert educational content creator tasked with generating factual questions and answers based on the following document excerpt. These questions should focus on retrieving specific details, figures, definitions, and key facts from the text.

Instructions:

- Generate **5** factual questions, each with a corresponding **expected_output**.
- Ensure all questions are directly related to the document excerpt.
- Present the output in the following structured JSON format:

[
  {
    "question": "What is the main purpose of the project described in the document?",
    "expected_output": "To develop a new framework for data security using AI-powered tools."
  },
  {
    "question": "Who authored the report mentioned in the document?",
    "expected_output": "Dr. Jane Smith."
  }
]
"""

# Generate dataset
import os
import json

dataset_file = 'qa_dataset.json'

if os.path.exists(dataset_file):
    # Load dataset from local file if it exists
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
else:
    # Generate dataset if local file doesn't exist
    dataset = []
    for doc in documents:
        qa_pairs = generate_qa(factual_prompt, doc.text, temperature=0.2)
        dataset.extend(qa_pairs)
    
    # Write dataset to local file
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f)

        
# Note: we're choosing to create the dataset in Langfuse below, but it's equally easy to create it in another platform.

from langfuse import Langfuse
langfuse = Langfuse()

dataset_name = "strategic_plan_qa_pairs"
langfuse.create_dataset(name=dataset_name);

for item in dataset:
  langfuse.create_dataset_item(
      dataset_name=dataset_name,
      input=item["question"],
      expected_output=item["expected_output"]
)
```

Go to the Langfuse dashboard, and confirm that the new dataset and dataset items have been added.

Now that we have an evaluation set, let's run our RAG pipeline and compare the RAG answer with the dataset. We'll use an LLM-as-a-judge to test for accuracy.

Create `evaluate_rag.py`
```python!
from langfuse import Langfuse
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

langfuse = Langfuse()

# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_evaluation(output, expected_output):
    client = openai.OpenAI()
    
    prompt = f"""
    Compare the following output with the expected output and evaluate its accuracy:
    
    Output: {output}
    Expected Output: {expected_output}
    
    Provide a score (0 for incorrect, 1 for correct) and a brief reason for your evaluation.
    Return your response in the following JSON format:
    {{
        "score": 0 or 1,
        "reason": "Your explanation here"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the accuracy of responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    result = eval(evaluation)  # Convert the JSON string to a Python dictionary
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

from datetime import datetime
 
def rag_query(input):
  
  generationStartTime = datetime.now()

  response = query_engine.query(input)
  output = response.response
  print(output)
 
  langfuse_generation = langfuse.generation(
    name="strategic-plan-qa",
    input=input,
    output=output,
    model="gpt-3.5-turbo",
    start_time=generationStartTime,
    end_time=datetime.now()
  )
 
  return output, langfuse_generation

def run_experiment(experiment_name):
  dataset = langfuse.get_dataset("strategic_plan_qa_pairs")
 
  for item in dataset.items:
    completion, langfuse_generation = rag_query(item.input)
 
    item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id
 
    score, reason = llm_evaluation(completion, item.expected_output)
    langfuse_generation.score(
      name="accuracy",
      value=score,
      comment=reason
    )

run_experiment("Experiment 2")
```

After running the experiment, navigate to your Dataset. Each experiment is called a "Run". See where your RAG output differed from the expected output.

### Milestone 8: Custom RAG pipeline (optional)

Instead of using the query engine, implement it manually. The high level steps for that are:

1. Fetch documents that have a high similary score with the query. 
   - In this context, remember that "documents" refer to chunks of text, often ~1k words, although this size is configurable
```python!
# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a retriever to fetch relevant documents
retriever = index.as_retriever(retrieval_mode='similarity', k=3)

# Define your query
query = "What years does the strategic plan cover?"

# Retrieve relevant documents
relevant_docs = retriever.retrieve(query)

print(f"Number of relevant documents: {len(relevant_docs)}")
print("\n" + "="*50 + "\n")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:")
    print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
    print(f"Metadata: {doc.node.metadata}")
    print(f"Score: {doc.score}")
    print("\n" + "="*50 + "\n")
```
2. Craft an LLM prompt that combines the documents with the query and generate a response.

Implementing this step manually gives you the freedom to add additional stages or logic to the RAG process.

## Capstone Project - Week 2

Last week, the project milestone was to select your 3 top app ideas with your group (or solo, if you're working alone).

This week, choose the app idea that you'd like to move forward with.

### Week 2 Capstone Milestones

- [ ] Create a public GitHub repo with a README.md that describes the project
- [ ] Create the app scaffold
   - [ ] Basic Python project with a .env
   - [ ] Tracing with LangSmith or Langfuse
   - [ ] Chainlit wired to OpenAI with chat history
- [ ] Design the main prompt(s)
- [ ] Create an evaluation dataset (10 examples)
- [ ] (stretch) Set up LlamaIndex loaders (e.g., email, Slack, wherever your data is coming from)
- [ ] (stretch) Set up your RAG pipeline
- [ ] (stretch) Run an LLM-as-a-judge evaluation test

## Submission

Run the lab, tinker around with it, make sure you understand the code, and upload to GitHub.

Submit your [GitHub repos for Week 2 here](https://forms.gle/KUDA88mSeqXTNNXW9).
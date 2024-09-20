---
title: Week 1 Project - Practical LLM Bootcamp for Devs

---

# Practical LLM Bootcamp for Devs

## Week 1 - Project: Evaluation

Project ideas:

- Language coach. Design your take on how you would like to practice another language.
- Document summarizer. Create a cliff notes-like effective compression of a document.
- Cybersecurity vulnerability analyzer. Review code for potential cybersecurity issues.
- Consumer researcher. Review web content and recommend consumer products

### Milestone 1 - Develop a prompt

Choose one of the ideas above (note: this is different from your capstone). You can also choose another idea, but don't spend too much time, we're just practicing for now.

### Milestone 2 - Scaffold the project

1. Create a new Python project
2. Create a Chainlit app with history and your system prompt
3. Set up Langsmith and traces

### Milestone 3 - Create an evaluation dataset

Use your app, and identify 10 representative interactions to add to an evaluation dataset.

A representative interaction is any common, happy path interaction you expect the prompt to be able to support. You can also add common edge cases.

### Milestone 4 - Identify 2 key metrics

Identify 2 key metrics for a quality response, and set up an LLM-as-a-judge to score them.

For example, let's say you chose the consumer research bot above. First, you can assume that your AI solution will have access to one (or many) articles. For the real solution, you would have a crawler element that intakes information from Google, Reddit, Amazon, etc. For an initial MVP, you can choose just Google.

Let's say that you're looking for projectors, and you use this article as input: https://www.pcmag.com/lists/best-projectors

If you were a consumer researcher, how would you process this?

Minimally, I would expect:
- Successful extraction. It's able to pull out the products with product info (ratings, link, pros / cons, etc)
- Source quality rating. How legitimate and reliable is this article

To recap, in this example, you would copy an article into the dataset input, and your output would be whatever you expected from your prompt (e.g., a ranked list of products).

### Milestone 5 - Deploy

Deploy on Render (https://render.com/)

- Click + New -> Web service
- Add your GitHub or git url
- Start command
  - `chainlit run app.py -h`
- Add your OpenAI key in the environment variables

### Milestone 6 - Capstone Project

- If relevant, meet with your group, and decide on your top 3 favorite app ideas.

### Milestone 7 (optional) - Run evaluation on Mistral

Run the inputs in your dataset on Mistral instead of OpenAI, and compare evaluations from OpenAI and Mistral.

### Milestone 8 (optional) - Expand dataset to 100

Use a series of prompts and temperature settings to create a balanced set of 100 high quality examples.

Being able to generate high quality examples is the precursor to effective fine tuning.

## Submission

After pushing your code to GitHub, submit your repository by filling out the [form here](https://docs.google.com/forms/d/e/1FAIpQLScW-sN6vQGcoHUg3OS6j8lruOd82meIHMuFLr6SG8jugxDX3A/viewform?usp=sf_link). Make sure the repo is publicly visible.

## Team and my contribution

A small 2-person project with Nika Soleimanzadeh. I worked on one version of the RAG notebook; most of the repository was my teammate's work. I keep it here as part of my learning, with credit to her.

---

# Marbet RAG Chatbot - Team Nika & Ali

This project is our submission for the Marbet Week 9/10 Challenge. It is a chatbot
that answers guest questions about Marbet's incentive trips. It uses RAG
(Retrieval-Augmented Generation), so instead of guessing it looks up the answer in
the event documents first and then writes a reply based on what it found. It can
answer questions about schedules, activities, what to pack, ship services, and
travel rules.

It uses LangChain to tie things together, FAISS to search the documents, and
Ollama (the llama3.2 model) to write the answers.

This was a two person project (Nika and Ali). I worked on the notebook: reading
the PDFs, splitting them into chunks, building the FAISS store, and getting the
RAG chain to give answers grounded in the documents.

## What It Does

- Answers natural language questions like:
  - "What are the activities on Day 3?"
  - "Do I need a visa?"
  - "What are the spa rules?"
- Retrieves information from the following documents:
  - Activities Overview
  - Packing List
  - Scenic Eclipse A-Z Guide
  - Tutorials and Travel Info
- Uses a FAISS vector store for semantic search
- Generates context-aware responses using Ollama's llama3.2 model

## Tech Stack

- Python
- LangChain
- FAISS
- Ollama (llama3.2)
- Jupyter Notebook (for development and testing)

## How to Run

1. Connect to the BUas VPN if off-campus.  
   Ollama is hosted on the BUas server and requires VPN access.  
   VPN setup instructions: [BUas VPN Guide](https://adsai.buas.nl/Study%20Content/Data%20Engineering/assets%2FADSAI%20Remote%20VPN.pdf)

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Put the event PDFs in a `documents` folder next to the notebook.
   The notebook reads every PDF from that folder.

4. Open `marbet_chatbot.ipynb` in Jupyter and run the cells in order.
   The first cells read the PDFs, split them into chunks, and build the
   FAISS vector store. The last cell starts a loop where you can type a
   question and get an answer. Type `exit` to stop.

Note: the chatbot uses the Ollama `llama3.2` model. The notebook expects
Ollama at `http://localhost:11434`. The original project used Ollama on the
BUas server, so change the `base_url` in the notebook if you run Ollama
somewhere else.


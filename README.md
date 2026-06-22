## Team and my contribution

A small 2-person project with Nika Soleimanzadeh. I worked on one version of the RAG notebook; most of the repository was my teammate's work. I keep it here as part of my learning, with credit to her.

---

# Marbet RAG Chatbot - Team Nika & Ali

This is our submission for the Marbet Week 9/10 Challenge. Marbet runs incentive
trips, and guests have a lot of small questions before and during the trip: what
is on the schedule, what to pack, what the ship services are, whether they need a
visa. All of that is already written down in the event PDFs, but nobody wants to
read four documents to find one answer. So we built a chatbot that reads the
documents for you and answers the question in plain language.

It uses RAG (Retrieval-Augmented Generation). Instead of letting the model guess
from memory, the chatbot first searches the event documents for the parts that
match your question, and only then writes an answer based on those parts. This
keeps the answers tied to the real Marbet documents and not to whatever the model
made up.

The notebook variant in this repo was my part. I worked on reading the PDFs,
splitting them into chunks, building the FAISS search index, and wiring the
RetrievalQA chain so the answers come from the documents.

## What it does

You ask a question in normal language and it answers from the event documents.
Things it can answer:

- "What are the activities on Day 3?"
- "Do I need a visa?" / "What document must you upload during an ESTA or eTA application?"
- "What should I pack?"
- "What are the spa rules?" or other ship service questions

The documents it reads come from the Marbet event pack:

1. Activities and excursions
2. Packing list
3. Scenic Eclipse A-Z Guide (the ship guide)
4. Tutorials and additional travel info
5. Travel itinerary

It runs as a small loop in the notebook: it keeps asking for a question and
printing an answer until you type `exit`.

## How it works (my approach)

The notebook goes through these steps in order.

**1. Read the PDFs.** I wrote `extract_text_from_pdfs(folder_path)` using PyMuPDF
(`fitz`). It loops over every file in the `documents` folder, skips anything that
is not a `.pdf`, opens each PDF, and reads the text page by page into one string
per document. I loop over the folder instead of listing files by hand so we do
not have to edit code when a document is added. Early on I had hard-coded my own
Desktop path and it broke for Nika, so I switched to a relative `documents`
folder that works on any machine.

**2. Split the text into chunks.** The full PDF text is too long to send to the
model at once, so I cut it with `RecursiveCharacterTextSplitter`. I first tried
`chunk_size=100` and the chunks were too small, the answers had almost no context.
I moved to `chunk_size=800` with `chunk_overlap=120` and that worked much better.
The overlap keeps a bit of the previous chunk so a sentence that gets cut at a
chunk boundary is not lost. The overlap has to be smaller than the chunk size or
the splitter errors out.

**3. Embed and build the FAISS index.** I turn each chunk into a vector with
`OllamaEmbeddings(model="llama3.2")`, the same llama3.2 model I use for the
answers. Then I build a FAISS vector store with `FAISS.from_texts`. FAISS is the
search part: for a question it finds the chunks that are closest in meaning,
instead of me sending all the document text to the model every time. I save the
store to disk with `save_local("marbet_vectorstore")` so I do not have to
re-embed all the PDFs on every run.

**4. Set up the model and the prompt.** The answers come from `ChatOllama` with
`model="llama3.2"`, pointed at Ollama through `base_url`. The retriever
(`vectorstore.as_retriever()`) pulls the matching chunks. My prompt tells the
model to answer from the provided context, and it lists the five document types
so the model knows what kind of material it is looking at. `{context}` gets the
retrieved chunks and `{question}` gets the user question. I started with a shorter
prompt without the document list, but the longer one with the list gave better
answers, so I kept it. (I left the old prompt in the notebook as a comment so I
could remember the difference.)

**5. Tie it together with RetrievalQA.** I use
`RetrievalQA.from_chain_type(llm, retriever, chain_type_kwargs={"prompt": prompt},
return_source_documents=True)`. So one call retrieves the chunks, fills the
prompt, and asks the model. I set `return_source_documents=True` so I could check
which chunks an answer actually came from while I was testing.

To check each piece worked, I tested the retriever on its own first (printing the
chunks it returned for the ESTA/eTA visa question, no model yet), then a one-shot
cell that shows the chunks and the answer side by side, then the final chat loop.

## Tech stack

- Python
- LangChain (`RecursiveCharacterTextSplitter`, `RetrievalQA`, `ChatPromptTemplate`)
- FAISS (`faiss-cpu`) for the vector search
- Ollama with the `llama3.2` model, used for both embeddings and chat
- PyMuPDF (`pymupdf` / `fitz`) for reading the PDFs
- Jupyter Notebook for development and testing

## Results and limitations

For the questions in the documents (activities, packing, visa/ESTA, ship
services), the chatbot answers from the right chunks. The retriever-only test on
the ESTA/eTA question returned the relevant visa chunks, which is what showed me
the retrieval part was working before I added the model on top.

I did not run a formal evaluation with accuracy numbers. I checked answers by hand
against the source documents, which is why I kept `return_source_documents=True`.
Honest limitations:

- The answer quality depends on the chunking. With `chunk_size=800` the answers
  were good for most questions, but for an answer that is spread across many
  pages the retrieved chunks can miss part of it.
- Using llama3.2 for both embeddings and chat is simple, but a dedicated
  embedding model would probably retrieve better.
- It needs the Ollama service running and the PDFs present; it does not work
  offline on its own.

## How to run

1. Connect to the BUas VPN if you are off-campus.
   Ollama was hosted on the BUas server and needs VPN access.
   VPN setup: [BUas VPN Guide](https://adsai.buas.nl/Study%20Content/Data%20Engineering/assets%2FADSAI%20Remote%20VPN.pdf)

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Put the event PDFs in a `documents` folder next to the notebook. The notebook
   reads every PDF from that folder, so the documents are not committed here.

4. Make sure Ollama is running with the `llama3.2` model pulled. The notebook
   uses `base_url="http://localhost:11434"`. The original project used Ollama on
   the BUas server, so change `base_url` in the notebook if you run Ollama
   somewhere else.

5. Open `marbet_chatbot.ipynb` in Jupyter and run the cells in order. The first
   cells read the PDFs, split them into chunks, and build the FAISS store. The
   last cell starts the loop where you type a question and get an answer. Type
   `exit` to stop.

## Key files

- `marbet_chatbot.ipynb` - the full notebook: PDF reading, chunking, FAISS build, and the RetrievalQA chat loop
- `requirements.txt` - the Python dependencies
- `Marbet report.pdf` - the project report
- `marbet_vectorstore/` - the saved FAISS index (created when you run the notebook)

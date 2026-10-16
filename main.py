import createDoc

from openai import OpenAI
from sentence_transformers import SentenceTransformer


client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key="#")


def retrieve(query,chunks_all):
  k=3
  model = SentenceTransformer("all-MiniLM-L6-v2")
  query_embedding = model.encode([query])
  distances,indices = index.search(query_embedding,k)

  results = []
  for i in indices[0]:
    results.append(chunks_all[i])

  return results

def build_prompt(question,contexts):
  context = "\n\n".join(contexts)

  prompt = f"""
  You are a question answering assistant , answer the questions using only context
  below. first introduce yourself as assistant jack and greet the user.
  if the answer is not contained in the context, say that the information is not available.

  context :

  {context}


  question:

  {question}

  """


  return prompt

docs = createDoc.create_docs()
print(docs)
index,chunks_all = createDoc.embedDoc(docs) 

question = "What is the name of the company"

contexts = retrieve(question,chunks_all)

prompt = build_prompt(question,contexts)

response = client.chat.completions.create(model="poolside/laguna-s-2.1:free",
                                          messages = [
                                              {"role":"user","content":prompt}
                                          ]
                                          )
print(response.choices[0].message.content)
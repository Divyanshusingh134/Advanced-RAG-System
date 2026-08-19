import os
import chromadb
import logging
import httpx
import hashlib
import math
import asyncio
import re
from nltk import sent_tokenize
from dotenv import load_dotenv
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from sentence_transformers import CrossEncoder
#----Get the api key and the urls----
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from the environment")

embedding_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
generation_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"


noisy_loggers = [
    "httpx", 
    "httpcore", 
    "huggingface_hub", 
    "filelock", 
    "urllib3", 
    "sentence_transformers", 
    "transformers"
]
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

class GeminiClient:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries
        self.embedding_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
        self.generation_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
        self.headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    async def fetch_data(self, http_client: httpx.AsyncClient, chunk: str, index: int, total_chunks: int, semaphore: asyncio.Semaphore) ->list[float]:
        payload = {
            "model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": chunk}]}
        }

        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await http_client.post(url=self.embedding_url, json=payload)
                    response.raise_for_status()
                    raw_data = response.json()
                    if "embedding" not in raw_data or "values" not in raw_data["embedding"]:
                        raise ValueError(f"Unexpected response schema: {raw_data}")
                    logging.info(f"Successfully generated embedding for chunk {index + 1}/{total_chunks}")
                    return raw_data["embedding"]["values"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 or e.response.status_code >= 500: 
                        wait_time = (2 ** attempt) + 1
                        logging.warning(f"Rate limited / Server error ({e.response.status_code}) on chunk {index + 1}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"HTTP error on chunk {index + 1}: {e}")
                        raise
            raise RuntimeError("Max retries exceeded due to rate limiting.")

    async def embed(self, http_client: httpx.AsyncClient, chunks: list[str], concurrency_limit: int = 5) -> list:
        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit)

        async with asyncio.TaskGroup() as tg:
            for i, chunk in enumerate(chunks):
                task = tg.create_task(self.fetch_data(http_client=http_client, chunk=chunk, index=i, total_chunks=len(chunks), semaphore=semaphore))
                tasks.append(task)

        return [task.result() for task in tasks]

            

class VectorDB:
    def __init__(self, storage_path = "./chroma_storage"):
        self.client = client = chromadb.PersistentClient(storage_path)

    def setup_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            configuration={
                "hnsw":{
                    "space":"cosine",
                    "ef_construction":200
                }
            },
            metadata={"description": "Cricket documentation vectors"}
        )

    def upsert_document(self, collection, chunks: list[str], vectors: list[list[float]], file_path: str):
        ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]

        chunk_metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents = chunks,
            metadatas = chunk_metadatas
        )


class TextProcessor:
    @staticmethod
    def sentence_window(file_path: str, chunks_size:int, overlap = 2) -> list:
        chunks = []
        if overlap >= chunks_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunks size ({chunks_size})")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError as e:
            logging.error(f"File not found at path: {file_path}")
            raise
        sentances = sent_tokenize(content)
        for i in range(0, len(sentances), chunks_size - overlap):
            chunks.append(" ".join(sentances[i: i + chunks_size]))
        return chunks

    @staticmethod
    def fixed_size_chunks(file_path :str, chunks_size :int, overlap=50) -> list:
        chunks = []
        if overlap >= chunks_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunks_size ({chunks_size})")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError as e:
            logging.error(f"File not found at path: {file_path}")
            raise
        words = content.split()
        for i in range(0, len(words), chunks_size - overlap):
            chunks.append(" ".join(words[i : i + chunks_size]))
        return chunks


class RAGPipeline:
    def __init__(self, api_key: str):
        self.llm = GeminiClient(api_key=api_key)
        self.db = VectorDB()

    
    async def run_evaluator(self, data_file: str, queries_file: str, output_file: str):
        HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=self.llm.headers) as http_client:

            collection_sentence_window = self.db.setup_collection("collection_sentence_window")
            collection_fixed_size = self.db.setup_collection("collection_fixed_size")

            if collection_sentence_window.count() == 0:
                sentence_window_chunks = TextProcessor.sentence_window(file_path=data_file, chunks_size=5, overlap=2)
                sentence_window_vectors = await self.llm.embed(http_client=http_client, chunks=sentence_window_chunks)
                self.db.upsert_document(collection_sentence_window, chunks=sentence_window_chunks, vectors=sentence_window_vectors, file_path=data_file)
                
            if collection_fixed_size.count() == 0:
                fixed_size_data = TextProcessor.fixed_size_chunks(file_path=data_file, overlap=50, chunks_size=200)
                fixed_size_vectors = await self.llm.embed(http_client=http_client, chunks=fixed_size_data)
                self.db.upsert_document(collection_fixed_size, chunks=fixed_size_data, vectors=fixed_size_vectors, file_path=data_file)

























# Set ChromaDB client
client = chromadb.PersistentClient("./chroma_storage")
# set the encoder 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# HTTP requests timeout 
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

# Generation for the result from the given context via Gemini API
async def generation(http_client: httpx.AsyncClient, retrieved_chunks: list[str], query: str, max_attempts: int = 3):
    context : str = "\n".join(f"Context {i + 1}: {s}" for i, s in enumerate(retrieved_chunks))
    # promt
    prompt: str = f"""You are a Question Answer Assistant who answers the question asked only from the given context.
    You are not allowed to use any extra knowledge and if there is no answer from the context you should just say honestly that its not found
    rather than making something up. Answer in complete sentences, not just a number or phrase
    Do not start your answer with phrases like 'from the given context' or 'based on the context'. Just Answer directly.
    The Context: {context},
    The Question: {query}"""

    payload = {
        "contents": [{
            "parts":[{"text": prompt}]
        }]
    }
    for attempt in range(max_attempts):
        try: 
            response = await http_client.post(url=generation_url,json=payload)
            response.raise_for_status()
            raw_data = response.json()
            candidate = raw_data.get("candidates")
            if not candidate:
                logging.warning(f"Gemini response contained no candidates")
                continue
            logging.info(f"Successfully Generated the Response for query {query}")
            return candidate[0]["content"]["parts"][0]["text"]
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 or e.response.status_code >= 500:
                wait_time = 35 + (10 * attempt)
                logging.warning(f"Rate Limited (429), Body: {e.response.text}")
                logging.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else :
                logging.error(f"HTTP error: {e.response.status_code} {e.response.text}")
                raise
        except httpx.RequestError as e:
            wait_time = 35 + (10 * attempt)
            logging.warning(f"Network error ({type(e).__name__}): {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
        except (KeyError, IndexError) as e:
            logging.error(f"Unexpected response shape: {e}")
            raise
    raise RuntimeError("Max retries exceeded due to rate limit.")

async def faithfulness_score(http_client: httpx.AsyncClient, retrieved_chunks: list[str], result:str, query:str, max_attempts: int = 3):
    context = "\n".join(f"context: {i + 1}: {s}" for i, s in enumerate(retrieved_chunks))
    # prompt
    prompt = f"""You are a score generate who generates a score only as 0 or 1, you would be looking at the context, query and the result I provide you, and generate a score on faithfullness.
    If the the answer contain only and only information from the Context (retrived_chunks) I provided then the score would be 1 and If something was invented then 0. 
    The response should only be an integer, no explantion no quotation and no extra work just one integer 0 or 1.
    Context: {context},
    Query: {query},
    Answer: {result}"""

    payload = {
         "contents": [{
            "parts":[{"text": prompt}]
        }]
    }
    for attempt in range(max_attempts):
        try:
            response = await http_client.post(url=generation_url, json=payload)
            response.raise_for_status()
            raw_data = response.json()
            candidate = raw_data.get("candidates")
            if not candidate:
                logging.warning(f"Gemini response contained no candidates")
                continue
            logging.info(f"Successfully Generated the Response for query: {query}")
            text = candidate[0]["content"]["parts"][0]["text"]
            try:
                score = int(text)
            except ValueError:
                match = re.search(r"\d+", text)
                score = int(match.group()) if match else 0
            return score
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 35 + (10 * attempt)
                logging.warning(f"Rate Limited {e.response.status_code}, Body: {e.response.text}")
                logging.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            if e.response.status_code >= 500:
                wait_time = 60 + (20 * attempt)
                logging.warning(f"Rate Limited {e.response.status_code}, Body: {e.response.text}")
                logging.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"HTTP error: {e.response.status_code} {e.response.text}")
                raise
        except httpx.RequestError as e:
            wait_time = 35 + (10 * attempt)
            logging.warning(f"Network error ({type(e).__name__}): {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
        except (KeyError, IndexError) as e:
            logging.error(f"Unexpected Response Shape: {e}")
            raise
    raise RuntimeError("max retries exceeded due to rate limit.")



async def relevance_score(result: str, query: str):
    pair = [[query, result]]
    raw_score = reranker.predict(pair, show_progress_bar=False)[0]
    normalized_score = 1 / (1 + math.exp(-raw_score))
    return round(normalized_score, 2)


async def re_ranking(retrieved_chunks: list[str], query: str, top_k:int = 3) -> list[str]:
    # loop through evry query
    if not retrieved_chunks:
        return []
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]

# Get the vectors from gemini embedding model
async def fetch_data(http_client: httpx.AsyncClient, chunk: str, index : int, total_chunks: int, semaphore: asyncio.Semaphore, max_retries: int = 3) -> list[float]:
    payload = {
        "model": "models/gemini-embedding-2",
        "content": {
        "parts": [{
            "text": chunk
        }]
        }
    }
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await http_client.post(url=embedding_url, json=payload)
                response.raise_for_status()
                raw_data = response.json()
                if "embedding" not in raw_data or "values" not in raw_data["embedding"]:
                    raise ValueError(f"Unexpected response schema: {raw_data}")
                logging.info(f"Successfully generated embedding for chunk {index + 1}/{total_chunks}")
                return raw_data["embedding"]["values"]
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500: 
                    wait_time = (2 ** attempt) + 1
                    logging.warning(f"Rate limited / Server error ({e.response.status_code}) on chunk {index + 1}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"HTTP error on chunk {index + 1}: {e}")
                    raise
        raise RuntimeError("Max retries exceeded due to rate limiting.")

#  Call function to get the vectors as TaskGroup by creating a courtine 
async def embed(http_client: httpx.AsyncClient, chunks: list, concurrency_limit: int = 5) -> list[list[float]]:
    tasks = []
    semaphore = asyncio.Semaphore(concurrency_limit)
    async with asyncio.TaskGroup() as tg:
        for i, chunk in enumerate(chunks):
            task = tg.create_task(fetch_data(http_client,chunk, i, total_chunks=len(chunks), semaphore=semaphore))
            tasks.append(task)
    return [task.result() for task in tasks]

# Get the chunks using sentance window with nltk 
def sentence_window(file_path: str, chunks_size:int, overlap = 2) -> list:
    chunks = []
    if overlap >= chunks_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunks size ({chunks_size})")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError as e:
        logging.error(f"File not found at path: {file_path}")
        raise
    sentances = sent_tokenize(content)
    for i in range(0, len(sentances), chunks_size - overlap):
        chunks.append(" ".join(sentances[i: i + chunks_size]))
    return chunks

# Get the chunks using fixed size window
def fixed_size_chunks(file_path :str, chunks_size :int, overlap=50) -> list:
    chunks = []
    
    if overlap >= chunks_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunks_size ({chunks_size})")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError as e:
        logging.error(f"File not found at path: {file_path}")
        raise
    words = content.split()
    for i in range(0, len(words), chunks_size - overlap):
        chunks.append(" ".join(words[i : i + chunks_size]))
    return chunks

# main
async def main():
    #  Create two seprate collection for the sentance window and fixed size
    collection_sentence_window = client.get_or_create_collection(
        name = "Cricket-docs-sentance-window",
        configuration={
            "hnsw":{
                "space":"cosine",
                "ef_construction":200
            }
        },
        metadata={"description": "Cricket documentation vectors"}
    )
    collection_fixed_size = client.get_or_create_collection(
        name = "Cricket_docs",
        configuration={
            "hnsw":{
                "space":"cosine",
                "ef_construction":200
            }
        },
        metadata={"description": "Cricket documentation vectors"}
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    # Start the async http_client
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=10.0, read=60.0), headers=headers) as http_client:
        # feed the data to the collections if they are empty 
        if collection_sentence_window.count() == 0:
            file_path =  'text.txt'
            # get chunks
            chunks = sentence_window(file_path=file_path, chunks_size=5, overlap=2)
            # get ids
            ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
            #  get metadats
            chunk_metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
            vectors = await embed(http_client=http_client, chunks=chunks)
            collection_sentence_window.upsert(
                ids=ids,
                embeddings=vectors, #type: ignore
                documents=chunks,
                metadatas=chunk_metadatas #type:ignore
            )

        if collection_fixed_size.count() == 0:
            # Get the file path
            file_path =  'text.txt'
            # fixed size
            chunks = fixed_size_chunks(file_path=file_path, chunks_size=200, overlap=50)
            # create the collection
            # Get the ids
            ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
            #  get metadata 
            chunk_metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
            vectors = await embed(http_client, chunks=chunks)
            collection_fixed_size.upsert(
                ids=ids,
                embeddings=vectors, #type:ignore
                documents=chunks,
                metadatas=chunk_metadatas #type:ignore
            )
        
        # Get the queries
        file_path = 'queries.txt'
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                 queries = file.readlines()
        except FileNotFoundError:
            logging.error(f"File not found at path: {file_path}")
            raise 
        
        query_vectors = await embed(http_client, queries)
        with open("eval_results.md", "a+") as file:
            file.write("## RAG Evaluation Results\n\n")
            stats = {
                "sentence": {"relevance" : 0.0, 
                             "faithful": 0},
                "fixed": {"relevance" : 0.0, 
                             "faithful": 0},
            }
            relevance_sum = 0
            faithful_count = 0
            for k, query in enumerate(queries):
                if not query_vectors or len(query_vectors) <= k:
                    logging.warning(f"Query vector for query {k} is missing")
                    continue
                # get the result for both 
                results_fixed_size = collection_fixed_size.query(
                    query_embeddings=[query_vectors[k]],
                    n_results=2
                )
                results_sentance_window = collection_sentence_window.query(
                    query_embeddings=[query_vectors[k]],
                    n_results=2
                )

                if not results_fixed_size["documents"] or not results_fixed_size["documents"][0]:
                    logging.warning(f"No document matches returned for query: {query}")
                    continue
                if not results_sentance_window["documents"] or not results_sentance_window["documents"][0]:
                    logging.warning(f"No document matches returned for query: {query}")
                    continue

                top_sentence = await re_ranking(retrieved_chunks=results_sentance_window["documents"][0], query=query)
                top_fixed = await re_ranking(retrieved_chunks=results_fixed_size["documents"][0], query=query)

                if not top_sentence:
                    logging.info(f"No Matched result found for Query: {query}")
                    continue
                if not top_fixed:
                    logging.info(f"No Matched result found for Query: {query}")
                    continue


                result_sentence = await generation(http_client, query=query, retrieved_chunks=top_sentence, max_attempts=3)
                result_fixed = await generation(http_client=http_client, query=query, retrieved_chunks=top_fixed)
                if result_sentence is None:
                    logging.error("Failed to extract task")
                    raise
                if result_fixed is None:
                    logging.error("Failed to extract task")
                    raise

                file.write(f"### Query {k + 1}: {query}\n\n")
                faith_score_sentence = await faithfulness_score(http_client=http_client, retrieved_chunks=results_sentance_window["documents"][0], query=query, result=result_sentence)
                faith_score_fixed = await faithfulness_score(http_client=http_client, retrieved_chunks=results_fixed_size["documents"][0], query=query, result=result_fixed)

                relevant_score_sentence = await relevance_score(query=query, result=result_sentence)
                relevant_score_fixed = await relevance_score(query=query, result=result_fixed)

                stats["fixed"]["relevance"] += relevant_score_fixed
                stats["fixed"]["faithful"] += faith_score_fixed
                stats["sentence"]["relevance"] += relevant_score_sentence
                stats["sentence"]["faithful"] += faith_score_sentence
                file.write(f"#### Strategy: Sentence Window\n")
                file.write(f"** Answer:** {result_sentence}\n")
                file.write(f"- **Relevance:** {relevant_score_sentence} | **Faithfulness:** {faith_score_sentence}\n\n")

                file.write(f"#### Strategy: Fixed Size\n")
                file.write(f"** Answer:** {result_fixed}\n")
                file.write(f"- **Relevance:** {relevant_score_fixed} | **Faithfulness:** {faith_score_fixed}\n\n")

                file.write("---\n\n")
                await asyncio.sleep(4)

            avg_rel_sent = stats["sentence"]["relevance"] / len(queries)
            avg_rel_fix = stats["fixed"]["relevance"] / len(queries)

            file.write("## Summary\n\n")
            file.write(f"- **Total Questions:** {len(queries)}\n\n")
            file.write("### Sentence-Window Strategy\n")
            file.write(f"- **Average Relevance Score:** {avg_rel_sent:.2f}\n")
            file.write(f"- **Faithfulness:** {stats['sentence']['faithful']}/{len(queries)} ({(stats['sentence']['faithful']/len(queries))*100:.0f}%)\n\n")
            
            file.write("### Fixed-Size Strategy\n")
            file.write(f"- **Average Relevance Score:** {avg_rel_fix:.2f}\n")
            file.write(f"- **Faithfulness:** {stats['fixed']['faithful']}/{len(queries)} ({(stats['fixed']['faithful']/len(queries))*100:.0f}%)\n")

            
        
if __name__ == "__main__":
    asyncio.run(main())


from fastapi import FastAPI, UploadFile, File
import utils
import hyperparams as hp
from contextlib import asynccontextmanager
import uvicorn
from prompts import get_prompt


# step-1: create a lifespan for the application
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # allocate resources
    # 1. load the LLM model
    llm = utils.get_llm()

    # 2. save the llm for application to use
    hp.llm = llm

    # run application
    yield
    # cleanup the resources
    hp.llm = None


# step-2: define the application
rag_app = FastAPI(title="Flash RAG", lifespan=app_lifespan)

# step-3: define routes
# [index, init_llm, upload_docs, query]


@rag_app.get("/")
def index():
    return {
        "message": "Hello! from Flash RAG application",
        "author": "Vinay Kumar",
        "year": 2025,
    }


@rag_app.get("/init_llm")
def init_llm():
    # TODO!
    kwargs = hp.model_args
    llm = utils.get_llm(**kwargs)
    hp.llm = llm
    return {"message": "LLM initialized", "LLM init params": kwargs}


@rag_app.post("/upload")
def upload_docs(
    docs: list[UploadFile] = File(...),
    collection_name: str = hp.vector_db_collection_name_default,
):
    docs_metadata = {}
    for doc in docs:
        # step-1: load/read every uploaded document
        fname = doc.filename
        fpath = f"../data/{fname}"
        try:
            contents = doc.file.read()
            with open(fpath, "wb") as f:
                f.write(contents)
            docs_metadata[fname] = {
                "status": "SUCCESS",
            }
        except Exception:
            docs_metadata[fname] = {
                "status": "FAILED",
            }
            return {"message": "Error reading the uploaded doc"}
        finally:
            doc.file.close()

        # step-2: tokenize various document types
        text_splitter = utils.get_text_splitter()
        if fname.endswith(".pdf"):
            doc_data = utils.preprocess_pdf_doc(
                fpath=fpath, text_splitter=text_splitter
            )
        else:
            return {"message": "Uploaded file-format NOT supported"}

        # step-3: create embeddings & store into the vector store
        utils.create_vector_db_collection(
            data=doc_data, collection_name=collection_name
        )

    return {
        "message": f"Successfully uploaded & stored {len(docs)} files to [{collection_name}] collection",
        "collection_name": collection_name,
        "docs_metadata": docs_metadata,
    }


@rag_app.get("/query")
def query(
    query: str,
    num_responses: int = hp.vector_db_num_responses,
    collection_name: str = hp.vector_db_collection_name_default,
):
    # step-1: get list of all collections
    collections = []
    try:
        with open(hp.vector_db_collections_fname, "r") as f:
            collections = f.read().split("\n")[:-1]
    except Exception:
        return {"message": "No collections found. Upload some documents first."}

    # step-2: if the queried collection is not found RETURN
    if collection_name not in collections:
        return {
            "message": f"The collection name = {collection_name} NOT found in the available collections list",
            "collection_name": collection_name,
        }

    # step-3: if the queried collection is available:
    # step-3a: - load the collection
    collection = utils.load_vector_db_collection(collection_name=collection_name)
    # step-3b: - retrieve the matching vector items to the query
    retrieved_responses = collection.query(query_texts=[query], n_results=num_responses)
    # step-3c: - create a prompt using the query & retrieved items
    prompt = get_prompt(user_prompt=query, retrieved_responses=retrieved_responses)

    # step-4: send prompt to the LLM & return the response
    llm_response = utils.call_llm(prompt=prompt)

    return {
        "user_query": f"{query}",
        "retrieved_docs": f"{retrieved_responses}",
        "llm_response": f"{llm_response}",
    }


# necessary to run the application via uvicorn
if __name__ == "__main__":
    uvicorn.run(app="main:rag_app", reload=True)

from fastapi import FastAPI
import utils
import hyperparams as hp
from contextlib import asynccontextmanager


# step-1: create a lifespan for the application
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # allocate resources
    # 1. load the LLM model
    llm = utils.get_llm()

    # 2. save the llm model for application to use
    hp.ml_models["llm"] = llm

    # run application
    yield
    # cleanup the resources
    hp.ml_models.clear()


# step-2: define the application
rag_app = FastAPI(title="Flash RAG", lifespan=app_lifespan)

# step-3: define routes
# [index, init_llm, upload_docs, query]


@rag_app.get("/")
def index():
    return "Hello! from Flash RAG application"


@rag_app.get("/init_llm")
def init_llm():
    pass


@rag_app.get("/query")
def query():
    pass


@rag_app.post("/upload")
def upload_docs():
    pass


if __name__ == "__main__":
    pass

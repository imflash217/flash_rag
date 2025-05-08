import chromadb
import hyperparams as hp
import langchain_text_splitters as lts
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import llamacpp

# from llama_cpp import Llama

chromadb_client = chromadb.PersistentClient(path=hp.vector_db_persistent_path)


def get_embeddings_model(model_name=hp.vector_db_embeddings_model_name):
    embeddings_func = (
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
    )
    return embeddings_func


def get_llm():
    llm = llamacpp.LlamaCpp(**hp.llm_model_args)

    # llm = Llama.from_pretrained(
    #     repo_id="TheBloke/nsql-llama-2-7B-GGUF",
    #     filename="nsql-llama-2-7b.Q2_K.gguf",
    #     **hp.llm_model_args,
    # )

    return llm


def call_llm(prompt=None):
    llm_response = hp.llm(prompt)
    # llm_response = None
    return llm_response


def get_text_splitter(
    chunk_size: int = hp.chunk_size, chunk_overlap: int = hp.chunk_overlap
):
    text_splitter = lts.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter


def preprocess_pdf_doc(fpath, text_splitter):
    data = PyPDFLoader(fpath).load_and_split(text_splitter)
    return data


def register_vector_db_collection(collection_name):
    # TODO! check if the collection already exists in the list
    #
    # APPEND a new collection to the list of all connections
    with open(hp.vector_db_collections_fname, "a") as f:
        f.write(collection_name + "\n")


def create_vector_db_collection(
    data,
    collection_name,
    client=chromadb_client,
):
    # step-1: check if the collection name is already available in the db
    # step-1: eventually get the collection object
    collections = client.list_collections()
    if len(collections) > 0 and collection_name in collections:
        # collection is already present
        collection = client.get_collection(name=collection_name)
    else:
        # collection not present so, create a new collection
        collection = client.create_collection(
            name=collection_name, embedding_function=get_embeddings_model()
        )

        # register the newly create collection's name to the vector db collections list
        register_vector_db_collection(collection_name=collection_name)

    # step-2: add data to the collection object
    num_embeddings = collection.count()
    num_data = len(data)

    collection.add(
        ids=[f"id_{i}" for i in range(num_embeddings, num_embeddings + num_data)],
        metadatas=[item.metadata for item in data],
        documents=[item.page_content for item in data],
    )


def load_vector_db_collection(
    collection_name,
    client=chromadb_client,
):
    collection = client.get_collection(name=collection_name)
    return collection

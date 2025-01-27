import hyperparams as hp
import langchain_text_splitters as lts


def get_prompt(user_prompt: str):
    prompt = f"""
{user_prompt}
Answer:
"""
    return prompt.strip()


def get_llm():
    return None


def get_text_splitter(
    chunk_size: int = hp.chunk_size, chunk_overlap: int = hp.chunk_overlap
):
    text_splitter = lts.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter


# TODO!
def preprocess_pdf_doc(fpath, text_splitter): ...
def preprocess_txt_doc(fpath, text_splitter): ...
def load_vector_db(data, collection_name): ...

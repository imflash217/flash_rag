llm = None
db_name = {}
llm_model_args = {
    "model_path": "../models/llama-2-7b.Q8_0.gguf",
    "n_gpu_layers": 500,
    "n_batch": 32,
    "max_tokens": 500,
    "n_ctx": 4096,
    "temperature": 0,
}
vector_db_embeddings_model_name = "all-MiniLM-L6-v2"
vector_db_collection_name_default = "flashrag_collection_default"
vector_db_num_responses = 2
vector_db_collections_fname = "COLLECTIONS.txt"
vector_db_persistent_path = "../data/vector_db_data"
chunk_size = 20
chunk_overlap = 10

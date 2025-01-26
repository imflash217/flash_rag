from fastapi import FastAPI
import utils
import hyperparams as hp


# step-1: create a lifespan for the application
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

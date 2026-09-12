"""
Company Policy RAG Agent — MLflow pyfunc wrapper.

This is the model that gets registered to Unity Catalog and (optionally) deployed as a
Databricks Model Serving endpoint. It's generated programmatically by
notebooks/policy_rag_agent.ipynb; this copy is kept here for readability and
version control. It shares its retrieval/prompting/citation logic with the notebook
through rag_core.py, which is bundled alongside this file at model-logging time via
mlflow.pyfunc.log_model(..., code_paths=["rag_core.py"]).
"""

import pickle
import json
import pandas as pd
import mlflow
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rag_core import build_rag_chain

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
MAX_NEW_TOKENS = 400


class RAGPolicyAgent(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # context.artifacts gives the paths to the vector DB and pickled chunks
        # that were bundled with this model when it was logged
        vector_db_path = context.artifacts["vector_db"]
        chunks_path = context.artifacts["doc_chunks"]

        # Reconnect to the persisted Chroma vector store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        vector_store = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)

        # Rebuild the BM25 keyword index from the pickled chunks
        with open(chunks_path, "rb") as f:
            doc_chunks = pickle.load(f)

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        bm25_retriever = BM25Retriever.from_documents(doc_chunks)
        bm25_retriever.k = TOP_K

        # Load the local LLM used to generate answers
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
            clean_up_tokenization_spaces=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        # Build the same hybrid-retrieval RAG chain used in the notebook
        self.rag_chain = build_rag_chain(vector_retriever, bm25_retriever, tokenizer, llm)

    def predict(self, context, model_input):
        # Accept a pandas DataFrame, a dict, or a plain string as input
        if isinstance(model_input, pd.DataFrame):
            query = model_input["user_message"].iloc[0]
        elif isinstance(model_input, dict):
            query = model_input.get("user_message", "")
        else:
            query = str(model_input)

        result = self.rag_chain.invoke(query)
        return {"response": result["answer"], "sources": json.dumps(result["sources"])}


# Tell MLflow which object to use as the servable model
mlflow.models.set_model(RAGPolicyAgent())

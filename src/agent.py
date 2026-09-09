"""
Company Policy RAG Agent — MLflow pyfunc wrapper.

This is the model that gets registered to Unity Catalog and (optionally) deployed as a
Databricks Model Serving endpoint. It's generated programmatically by
notebooks/policy_rag_agent.ipynb (Cell 4); this copy is kept here for readability and
version control.
"""

import pandas as pd
import mlflow
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT = (
    "You are a helpful company policy assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain the "
    "answer, state that you do not know."
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPolicyAgent(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # vector_db path comes from the bundled MLflow artifact, NOT a hardcoded /tmp path
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )
        vector_store = Chroma(
            persist_directory=context.artifacts["vector_db"],
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            do_sample=False,
            return_full_text=False,
            clean_up_tokenization_spaces=False
        )
        pipe.model.config.max_length = None
        llm = HuggingFacePipeline(pipeline=pipe)

        def build_prompt(inputs):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{inputs['context']}\n\nQuestion: {inputs['question']}"
                }
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(build_prompt)
            | llm
            | StrOutputParser()
        )

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            query = model_input["user_message"].iloc[0]
        elif isinstance(model_input, dict):
            query = model_input.get("user_message", "")
        else:
            query = str(model_input)
        return {"response": self.rag_chain.invoke(query)}


mlflow.models.set_model(RAGPolicyAgent())

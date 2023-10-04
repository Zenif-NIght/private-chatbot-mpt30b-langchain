
import os
import time

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS
# remove the db folder
import os 
import re
import pandas as pd
import tqdm
import subprocess
import sys

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
model_path = os.environ.get("MODEL_PATH")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))




def run_questions_list(question_list, llm):
    # Prepare the retriever
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs = {'device': 'cuda:0'})
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    return_list = {}
    # Interactive questions and answers over your docs
    for query in question_list:
        start_time = time.time()
        if query == "exit":
            break
        if query.strip() == "":
            print("Please enter a valid question.")
            continue

        # Get the answer from the chain
        try:
            print("Thinking... Please note that this can take a few minutes.")
            # start = time.time()
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]
            # end = time.time()

            # Print the result
            print("\n\n> Question:")
            print(query)
            # print(f"\n> Answer (took {round(end - start, 2)} s.):")
            print(answer)

            # Print the relevant sources used for the answer
            sources_used = ''
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
                # sources_used[f'{document.metadata["source"]}'] = document.page_content
                sources_used += f'{document.metadata["source"]}\n'
                sources_used += f'{document.page_content}\n\n'
            return_list[query] = answer +'\n\n\n'+ sources_used
            time_passed = round(time.time() - start_time, 2)    
            print(f"\n> Took {time_passed} s. or {round(time_passed / 60, 2)} min.")
        except Exception as e:
            print(str(e))
            raise
    return return_list





def load_model():
    try:
        # check if the model is already downloaded
        if os.path.exists(model_path):
            print("Loading model...")
            global llm
            # initialize llm
            llm = CTransformers(
                model=os.path.abspath(model_path),
                model_type="mpt",
                callbacks=[StreamingStdOutCallbackHandler()],
                config={"temperature": 0.1, "stop": ["<|im_end|>", "|<"]},
                # gpu layers
                gpu_layers=10_000_000,

            )
            return True
        else:
            raise ValueError(
                "Model not found. Please run `poetry run python download_model.py` to download the model."
            )
    except Exception as e:
        print(str(e))
        raise



def main():

    questions_list = ['What is the methodology used in this paper?','How does this paper improve dynamics robotic motion?']

    # cwd = os.getcwd()
    # try:
    #     subprocess.check_output([sys.executable, 'ingest.py'], 
    #                                                 cwd=cwd, 
    #                                                 )
    # except Exception as e:
    #     print(f'Error: {e}')
    #     return -1

    # load the model
    load_model()
    # run the questions
    resulty_dict = run_questions_list(questions_list, llm)

    
    def save_qa_dict_to_markdown(qa_dict, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for question, answer in qa_dict.items():
                # Write the question in markdown format
                file.write(f"### Q: {question}\n\n")
                
                # Write the answer in markdown format
                file.write(f"A: {answer}\n\n")


    save_qa_dict_to_markdown(resulty_dict, 'qa_output.md')



if __name__ == "__main__":
    main()




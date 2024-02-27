from fastapi import FastAPI, HTTPException # Framework for making python api 
from langchain.prompts import PromptTemplate # To pass chatbot the basic knowledge what task he should do
from langchain_core.output_parsers import JsonOutputParser # Format of output from the chatbot
from langchain_core.pydantic_v1 import BaseModel, Field # JSON framework
from langchain_openai import ChatOpenAI # LLM model of openai
import PyPDF2 # reading data from pdf
from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile # creating temporary file
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()
import os

# Replace 'YOUR_ENV_VARIABLE' with the name of the environment variable you want to retrieve
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Define FastAPI app
app = FastAPI()

# Initialize ChatOpenAI model
model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Function to read text from a PDF file
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Define data structure for CV extraction - this is structure in which llm going to extract data. Pass unstructed data and it will structed output.
class CVExtractor(BaseModel):
    name: str = Field(description="Name of the applicant")
    father_name: str = Field(description="Father of the applicant")
    phone_no: str = Field(description="Phone number of the applicant")
    address: str = Field(description="Address of the applicant")
    skills: List[str] = Field(description="Skills of the applicant")
    project: List[str] = Field(description="List of projects made by the applicant")
    programming_language: List[str] = Field(description="Programming language of the applicant")
    education_history: List[str] = Field(description="All info regarding applicant education")
    summary: str = Field(description="Summary of the applicant")

# Endpoint to extract information from CV
@app.post("/extract-cv-info/")
async def extract_cv_info(query: str = "Extract the useful info from the CV", file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            pdf_file_path = tmp.name

        # Read text from PDF
        text_from_pdf = read_pdf(pdf_file_path)

        # Set up parser
        parser = JsonOutputParser(pydantic_object=CVExtractor)
        # Creating the prompt template from this
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{text_from_pdf}\n",
            input_variables=["query", "text_from_pdf"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        #Creating the chain which is a common way to interact with the LLM.
        chain = prompt | model | parser

        # Invoke chain - inference with the LLM
        result = chain.invoke({"query": query, "text_from_pdf": text_from_pdf})
        
        # Remove the temporary file
        os.unlink(pdf_file_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

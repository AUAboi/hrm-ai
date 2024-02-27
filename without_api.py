from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import PyPDF2

model = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo",api_key="")

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # Initialize an empty string to store the text
        text = ""

        # Iterate through all pages and extract text
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_file_path = 'cv.pdf'
text_from_pdf = read_pdf(pdf_file_path)

# Define your desired data structure.
class cv_extracter(BaseModel):
    name: str = Field(description="name of the applicant")
    father_name: str = Field(description="father of the applicant")
    phone_no: str = Field(description="phone numbe of the applicant")
    adress: str = Field(description="adress of the applicant")
    skills: str = Field(description="skills of the applicant")
    project: str = Field(description="list of Project made by applicant")
    programming_language : str = Field(description="Programming languge in the text")
    education_history : str = Field(description="All info regarding applicant education")
    summary : str = Field(description="Summary of the applicant")
    

# And a query intented to prompt a language model to populate the data structure.
query = "Extract the useful info from the cv"
text = text_from_pdf
# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=cv_extracter)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n{text_from_pdf}\n",
    input_variables=["query","text_from_pdf"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

print(chain.invoke({"query": query,"text_from_pdf":text}))



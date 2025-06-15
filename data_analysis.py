from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from pydantic import BaseModel,Field
class CodeOutput(BaseModel):
    text: str = Field(description="This contains the textual part of the response")
    code: str = Field(description="This contains the complete code generated in python")

code_parser = PydanticOutputParser(pydantic_object=CodeOutput)
text_parser = StrOutputParser()
class DataInsights:
    def __init__(self,llm):
        self.llm = llm
        self.stats = PromptTemplate(
            template="Give me the descriptive Stat for the given text data without showing calculations\n{context}",
            input_variables=["context"]
        )

        self.missing = PromptTemplate(
            template="Do a missing value analysis on the given text\n {context}",
            input_variables=["context"]
        )

        self.correlation = PromptTemplate(
            template="""
Generate code for getting the correlation heatmap graph in this format for the given text in strictly json format
format:
{format}
text:
{context}
""",
            input_variables=["context"],
            partial_variables={"format": code_parser.get_format_instructions()}
        )

        self.outlier = PromptTemplate(
            template="""
Find out in which column there is an outlier then 
Generate code for getting the boxplot graph using iqr to detect outlier just generate the code for the boxplot only
put the details about the outlier  in the text key and the code for graph in the code key
in the given  format for the given text in strictly json format
format:
{format}
text:
{context}
""",
            input_variables=["context"],
            partial_variables={"format": code_parser.get_format_instructions()}
        )

        self.dtype = PromptTemplate(
            template="Give the datatype of all the given columns from this {context}",
            input_variables=["context"]
        )

    def run_stats(self, context):
        return (self.stats | self.llm | text_parser).invoke({"context": context})

    def run_missing(self, context):
        return (self.missing | self.llm | text_parser).invoke({"context": context})

    def run_correlation(self, context):
        return (self.correlation | self.llm | code_parser).invoke({"context": context})

    def run_outlier(self, context):
        return (self.outlier | self.llm | code_parser).invoke({"context": context})

    def run_dtype(self, context):
        return (self.dtype | self.llm | text_parser).invoke({"context": context})
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from pydantic import BaseModel, Field

class CodeOutput(BaseModel):
    text: str = Field(description="This contains the textual part of the response")
    code: str = Field(description="This contains the complete code generated in python")

code_parser = PydanticOutputParser(pydantic_object=CodeOutput)
text_parser = StrOutputParser()

class Summarizer:
    def __init__(self,llm):
        self.prompt = PromptTemplate(
            template="""
You are a helpful assistant. Summarize the following document:

{document}
""",
            input_variables=["document"]
        )
        self.llm = llm
        self.chain = self.prompt | llm | text_parser

    def run(self, document):
        return self.chain.invoke({"document": document})


class QnA:
    def __init__(self,llm):
        self.prompt = PromptTemplate(
            template="""
You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )
        self.llm = llm
        self.chain = self.prompt | llm | text_parser

    def run(self, context, question):
        return self.chain.invoke({"context": context, "question": question})


class CodeGenerator:
    def __init__(self,llm):
        self.prompt = PromptTemplate(
            template="""
You are an expert Python assistant. Return a valid JSON object only. Follow this format:
{instructions}
Context:
{context}
Prompt:
{prompt}
""",
            input_variables=["context", "prompt"],
            partial_variables={"instructions": code_parser.get_format_instructions()}
        )
        self.llm = llm
        self.chain = self.prompt | llm | code_parser

    def run(self, context, prompt):
        return self.chain.invoke({"context": context, "prompt": prompt})




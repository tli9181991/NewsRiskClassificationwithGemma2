import os
# from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
import pandas as pd

# llm = ChatOllama(model="gemma3:12b")
llm = ChatOllama(model="deepseek-r1")

def llm_labling(headline, description):
    
    prompt = PromptTemplate.from_template(
        """
        You are a professional assistant for financial risk management. 
        You will be given the headline and the description of a news.
        Analysis the risk of the news of the article reflecting.
        \n\n{headline}
        \n\n{description}
        Classify the risk of the news belong to and list your result.
        If there is no risk, say no risk.
        Do not add any descriptions and reasons in your answer.
        """
    )
    labeling_chain = (
        prompt
        |llm
        |StrOutputParser()    
    )

    result = labeling_chain.invoke({'headline': headline, 'description':description})
    analysis = result.split('/think>\n')[-1]
    analysis = analysis.replace('\n', ' ')

    return analysis

if __name__ == '__main__':
    df_news = pd.read_csv("./data/yf_news_NVDA.csv")

    risks = []
    for headline, description in zip(df_news['headline'], df_news['description']):
        inspected_risk = llm_labling(headline, description)

        risks.append(inspected_risk)
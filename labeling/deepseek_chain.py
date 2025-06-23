import os
# from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
import pandas as pd

llm = ChatOllama(model="deepseek-r1")

def deepseek_labling(headline, description):
    
    prompt = PromptTemplate.from_template(
        """
        You are a professional financial agent for risk management. 
        You will be given the headline and the description of a news.
        Analysis the risk of the news of the article reflecting.
        \n\n{headline}
        \n\n{description}
        Just tell the type of risk you have classified. If there is no risk, say no risk.
        """
    )
    labeling_chain = (
        prompt
        |llm
        |StrOutputParser()    
    )

    result = labeling_chain.invoke({'headline': headline, 'description':description})

    return result.split('/think>\n')[-1]

if __name__ == '__main__':
    df_news = pd.read_csv("./data/yf_news_NVDA.csv")

    risks = []
    for headline, description in zip(df_news['headline'], df_news['description']):
        inspected_risk = deepseek_labling(headline, description)

        risks.append(inspected_risk)
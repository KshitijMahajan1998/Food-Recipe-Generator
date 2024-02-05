from tabnanny import verbose
import dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from loomie_core.prompts import PROMPT_SUFFIX, _DEFAULT_TEMPLATE, SYSTEM
from loomie_core.similarity_search import fetch_similar_records
from loomie_core.logging_system import LoomieInteractionLog
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from sqlalchemy.exc import DatabaseError
from databricks.sql.exc import ServerOperationError
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()

class BlSQL:
    def __init__(self, db, session_data=None):
        self.db = db
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0,verbose=True)
        self.memory = ConversationBufferMemory()
        self.memory2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.memory_count = 0
        self.logger = LoomieInteractionLog()

        if session_data:
            self.logger.update_log("session_id", session_data.session_id)
            self.logger.update_log("user_info", session_data.user_info)
            self.logger.update_log("customer_code", session_data.customer_code)

    def get_prompt1(self,question):
        similar_records=[]
        similar_records=fetch_similar_records(question)
        PROMPT = PromptTemplate.from_template(
            _DEFAULT_TEMPLATE + ' '.join(similar_records) + PROMPT_SUFFIX,
            )
        return PROMPT

    def _set_SQLchain_executor(self,question):
        return SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            prompt=self.get_prompt1(question),
            verbose=True,
            memory=self.memory,
            use_query_checker=True,
            top_k=3,
        )
    
    def get_prompt_second(self,answer):
        content=SYSTEM+"Data given by restaurant company:"+answer
        PROMPT2 = ChatPromptTemplate.from_messages(
        [
        SystemMessage(
            content=content
        ), 
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ), 
        ])
        return PROMPT2

    def _set_LLMChain_executor(self,answer):
        return LLMChain(
            llm=self.llm,
            prompt=self.get_prompt_second(answer),
            memory=self.memory2,
            verbose=True,
            )

    def record_memory(self, question, response):
        self.memory_count += 1
        self.memory.chat_memory.add_user_message(
            f"QUESTION {self.memory_count}: {question}"
        )
        self.memory.chat_memory.add_ai_message(
            f"RESPONSE {self.memory_count}: {response}"
        )

    def _run_sql_chain(self, question) -> str:
        
        db_chain = self._set_SQLchain_executor(question)
        
        answer=""
        try:

            db_chain.run(question)

        except DatabaseError as e:

            answer="We don't have enough information"
        if answer=="":
            answer=str(self.memory.chat_memory.messages[-1]).split('=')[-1].strip()
        
        return(answer)
    
    def _run_llm_chain(self,question):
        input_datetime = datetime.now()
        self.logger.update_log("input_datetime", datetime.now())
        self.logger.update_log("user_input", question)
        sql_output=self._run_sql_chain(question)
        chat_llm_chain=self._set_LLMChain_executor(sql_output)
        chat_llm_chain.predict(human_input=question)
        response = str(self.memory2.chat_memory.messages[-1]).split('=')[-1].strip()
        self.record_memory(question, response)

        self.logger.update_log("model_response", response)

        end_datetime = datetime.now()
        self.logger.update_log(
            "model_response_time_ms",
            int((end_datetime - input_datetime).total_seconds() * 1000),
        )
        self.logger.insert_log_entry()

        return response



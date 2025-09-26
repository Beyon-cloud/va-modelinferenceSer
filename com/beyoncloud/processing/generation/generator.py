import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
from datetime import date
from datetime import datetime
import com.beyoncloud.config.settings.env_config as config
from typing import List, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from com.beyoncloud.models.model_loader import ModelRegistry
from com.beyoncloud.schemas.rag_reqres_data_model import (
    ChatHistory, RagReqDataModel, 
    StructureInputData, UserInput,
    RagLogQryModel, RagLogQryModelBuilder
)
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from com.beyoncloud.processing.prompt.prompt_template import (
    get_prompt_template,get_prompt_param, 
    get_prompt_input_variables,build_prompt,build_universal_prompt, 
    build_universal_prompt1, get_json_extraction_prompt,
    SafeDict, get_temp_prompt_template
)
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.processing.prompt.prompt_processing import PromptGenerator
from com.beyoncloud.utils.file_util import TextLoader
from com.beyoncloud.common.constants import PromptType, CommonPatterns
from com.beyoncloud.processing.prompt.prompt_generation.huggingface_llama_connect import HuggingFaceLlama3Client
from com.beyoncloud.db.postgresql_impl import PostgresSqlImpl
from com.beyoncloud.db.postgresql_connectivity import PostgreSqlConnectivity

logger = logging.getLogger(__name__)

class RagGeneratorProcess:
    """
    A class responsible for generating LLM responses using LangChain's Conversational Retrieval Chain.

    This class builds contextual answers by combining retrieved documents, user queries, and chat history.
    It supports dynamic prompt templating based on domain and response type configurations.

    Author: Jenson
    Date: 16-June-2025
    """

    def __init__(self):
        """
        Initializes an instance of the RagGeneratorProcess.

        The class assumes that the language model (LLM) is preloaded into the `ModelRegistry.hf_llama_model_pipeline`.
        """
        self.prompt_generator = PromptGenerator()

    async def generateAnswer(self, ragReqDataModel: RagReqDataModel, searchResults: List[Dict[str, Any]]):
        """
        Generates a response using LangChain's prompt templating and a loaded LLM model.

        This method combines the user query, top retrieved document chunks (`searchResults`), 
        and optional chat history to create a context-aware input for the model. 
        The prompt template and required input variables are fetched dynamically based on the 
        request's domain and response type.

        Args:
            ragReqDataModel (RagReqDataModel): Object containing metadata such as domain ID and response type.
            query (str): The current user question or query.
            searchResults (List[Document]): List of LangChain `Document` objects retrieved from the vector store.
            chat_history (List[ChatHistory], optional): List of past query-response pairs for conversational context.

        Returns:
            str: The generated answer from the LLM based on the provided inputs.
        
        Raises:
            ValueError: If the LLM model is not initialized in the `ModelRegistry`.
        """
        

        # Ensure LLM model is properly initialized
        allModelObjects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = allModelObjects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")

        query = self.getQuery(ragReqDataModel)
        logger.info(f"Inpur query --> {query}")
        
        # Build chat history string (if available)
        historyPrompt = []
        chat_history: List[ChatHistory] = self.getChatHistory(ragReqDataModel)
        if chat_history:
            for chat in chat_history[-5:]:
                historyPrompt.append((chat.query, chat.response))


        # Combine context from search results
        fullContext =""
        list_context = []
        if searchResults:
            fullContext = "\n".join([result['chunk'] for result in searchResults])
            list_context = [result['chunk'] for result in searchResults]

        context = fullContext[:1000]  # safe first

        """
        customPrompt = get_prompt_template(ragReqDataModel.domain_id, ragReqDataModel.response_type)
        promptInputVariables = get_prompt_input_variables(ragReqDataModel.domain_id, ragReqDataModel.response_type)
        
        variable_map = {
            "query": query,
            "context": context,
            "chat_history": historyPrompt
        }
        # Dynamically build the inputs dictionary
        inputs = {var: variable_map[var] for var in promptInputVariables}
        print("Inputs : ",inputs)
        print(f"customPrompt - {customPrompt}")
        """
        # Prompt Generation
        
        try:
            final_prompt = self.prompt_generator.prompt_generator(query,ragReqDataModel.domain_id, searchResults, chat_history)
        
        except Exception as e:
            logger.error(f"❌ Error in demonstration: {e}")
            import traceback
            traceback.print_exc()

        #task="Answer the question"
        #answer_style="one short paragraph"
        #today=date.today().strftime("%Y-%m-%d")
        #top_k=len(list_context)
        #chat_prompt_template = build_universal_prompt1(
        #    task, 
        #    query, 
        #    list_context, 
        #    top_k, 
        #    answer_style,
        #    language="English", 
        #    max_tokens=256, 
        #    today=today)

        #print(f"chat_prompt_template --> {chat_prompt_template}")

        #rendered_prompt = customPrompt.format(
        #    inputs
        #)
        #response = llm.invoke(rendered_prompt)

        # ========== LLM CHAIN ==========
        # Type 1: (Depriciated in langchan, so use Type 2)
        #chain = LLMChain(llm=llm, prompt=customPrompt)
        #response = chain.run(inputs)

        # Type 2:
        #chain = customPrompt | llm
        #response = chain.invoke(inputs)

        # Type 3:
        llama_tokenizer = allModelObjects.get_llama_tokenizer()
        #chat_prompt = build_prompt(fullContext,query,True,llama_tokenizer)
        #print(f"chat_prompt - {chat_prompt}")
        #logger.info(f"chat_prompt --> {chat_prompt}")
        response = llm.invoke(final_prompt)
        #response = llm.invoke(chat_prompt_template)

        print("\n Response:\n", response)
        logger.info(f"Response --> {response}")



        #llmResponse = llm(customPrompt)[0]['generated_text'] # String prompt
        #response = llmResponse
        #if "[/INST]" in llmResponse:
        #    response = llmResponse.split("[/INST]")[-1].strip()
        return response

    async def hf_response(self, structure_input_data: StructureInputData):
        """
        Structured Response Generation
        """
        
        # Ensure LLM model is properly initialized
        allModelObjects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = allModelObjects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")


        # Combine context from search results
        fullContext = structure_input_data.context_data
        #if structure_input_data.source_path:
        #    text_loader = TextLoader()
        #    fullContext = text_loader.get_text_content(structure_input_data.source_path)
        print(f"fullContext -----------> {fullContext}")

        # Prompt Generation
        prompt_output = get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type 
        )
        final_prompt = prompt_output["prompt"]
        prompt_id = prompt_output["prompt_id"]
        input_variables = prompt_output["input_variables"]
        print(f"input_variables1 -->{prompt_id} -  {input_variables}")
        param_result = await get_prompt_param(prompt_id)
        print(f"param_result --> {param_result}")
        print(f"output_type : {structure_input_data.output_format}")


        llama3_client = HuggingFaceLlama3Client(
            api_key=config.HF_KEY,
            model_name=config.HF_MODEL_NAME
        )
        #isConnect = llama3_client.test_connection
        #print(f"Is COnnect ----- {isConnect}")
        # Test connection
        if not llama3_client.test_connection():
            raise ConnectionError("Failed to connect to Hugging Face Llama3 API")

        variable_map = {
            "context": fullContext,
            "output_type": structure_input_data.output_format
        }
        for result in param_result:
            param_key = getattr(result, "param_key", None)
            param_value = getattr(result, "param_value", None)
            #print(f"param_key --> {param_key} - {param_value}")
            if param_key is not None:
                variable_map[param_key] = param_value

        system_prompt = prompt_output["system_prompt_template"].format_map(SafeDict(variable_map))
        user_prompt = prompt_output["user_prompt_template"].format_map(SafeDict(variable_map))

        response = llama3_client.generate_sync(
                    system_prompt=prompt_output["system_prompt_template"],
                    user_prompt=user_prompt,
                    timeout=60,
                    temperature=0.1,
                    max_tokens=2048
                )

     
        print("\n Hugging face Response:\n", response)
        logger.info(f"Response --> {response}")

        return response

    async def generate_structured_response(self, structure_input_data: StructureInputData):
        """
        Structured Response Generation
        """
        
        # Ensure LLM model is properly initialized
        allModelObjects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = allModelObjects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")


        # Combine context from search results
        fullContext = structure_input_data.context_data
        #if structure_input_data.source_path:
        #    text_loader = TextLoader()
        #    fullContext = text_loader.get_text_content(structure_input_data.source_path)

        # Prompt Generation
        prompt_output = get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type 
        )
        final_prompt = prompt_output["prompt"]
        prompt_id = prompt_output["prompt_id"]
        input_variables = prompt_output["input_variables"]
        print(f"input_variables1 -->{prompt_id} -  {input_variables}")
        param_result = await get_prompt_param(prompt_id)
        print(f"param_result --> {param_result}")

        variable_map = {
            "context": fullContext,
            "output_type": structure_input_data.output_format
        }
        for result in param_result:
            param_key = getattr(result, "param_key", None)
            param_value = getattr(result, "param_value", None)
            #print(f"param_key --> {param_key} - {param_value}")
            if param_key is not None:
                variable_map[param_key] = param_value

        #print(f"variable_map --> {variable_map}")

        # Dynamically build the inputs dictionary
        inputs = {var: variable_map[var] for var in input_variables}
        print(f"Inputs : --> {inputs}")

        #final_prompt = get_json_extraction_prompt(fullContext, structure_input_data.prompt_type)
        print(f"final_prompt --> {final_prompt}")
        #response = llm.invoke(final_prompt)


        #inputs = {
        #    "context": fullContext
        #}
        #chain = LLMChain(llm=llm, prompt=final_prompt)
        #response = chain.run(inputs)

        # Build runnable sequence (prompt | llm)
        chain = final_prompt | llm
        # Invoke asynchronously
        response = await chain.ainvoke(
            inputs
        )

        print("\n Response:\n", response)
        logger.info(f"Response --> {response}")

        return response

    async def temp_hf_response(self, structure_input_data: StructureInputData,
        system_prompt: str,
        user_prompt: str                           
    ):
        """
        Structured Huggingface Response Generation
        """
        
        # Get context data
        fullContext = structure_input_data.context_data
        print(f"fullContext -----------> {fullContext}")

        # Prompt Generation
        prompt_output = get_temp_prompt_template(
            system_prompt, 
            user_prompt 
        )

        llama3_client = HuggingFaceLlama3Client(
            api_key=config.HF_KEY,
            model_name=config.HF_MODEL_NAME
        )

        # Test connection
        if not llama3_client.test_connection():
            raise ConnectionError("Failed to connect to Hugging Face Llama3 API")

        variable_map = {
            "context": fullContext,
            "output_type": structure_input_data.output_format
        }

        final_system_prompt = prompt_output["system_prompt_template"].format_map(SafeDict(variable_map))
        final_user_prompt = prompt_output["user_prompt_template"].format_map(SafeDict(variable_map))

        starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()
        response = llama3_client.generate_sync(
                    system_prompt=final_system_prompt,
                    user_prompt=final_user_prompt,
                    timeout=60,
                    temperature=0.1,
                    max_tokens=2048
                )
        endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        end_time = time.time()
        elapsed = end_time - start_time  # in seconds (float)
        
        print("\n Hugging face Response:\n", response)
        logger.info(f"Response --> {response}")

        # Save response data into query log table
        try:
            metadata = {
                    "starttime": starttime,
                    "start_time": start_time,
                    "endtime": endtime,
                    "end_time": end_time,
                    "elapsed": elapsed,
                    "file_path": structure_input_data.source_path
            }

            rag_log_qry_model = (RagLogQryModelBuilder()
                .with_orgId(structure_input_data.organization_id)
                .with_query(CommonPatterns.EMPTY_SPACE)
                .with_response(response)
                .with_search_result_json([fullContext])
                .with_time_elapsed(elapsed)
                .with_metadata(metadata)
                .build()
            )

            await self.save_model_response(rag_log_qry_model)
        except Exception as e:
            logger.error(f"Error processing RAG query log save : {e}")

        return response

    async def save_model_response(
        self, 
        rag_log_qry_model = RagLogQryModel
    ):
        """
        Saves RAG responses as log entry in the database.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG request data.
            response : The RAG response string data.

        """
        
        pg_conn = PostgreSqlConnectivity()
        column_data = {
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "org_id"): rag_log_qry_model.orgId,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "qry_txt"): rag_log_qry_model.query,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "resp_txt"): rag_log_qry_model.response,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "top_results"): rag_log_qry_model.search_result_json,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "resp_time"): rag_log_qry_model.time_elapsed,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "metadata"): rag_log_qry_model.metadata,

        }
        RagQryLogs = pg_conn.Base.classes[self.table_settings.get_db_table_name("schema1", "RagQryLogs")]
        qry_log = RagQryLogs(**column_data)
        pg = PostgresSqlImpl()
        await pg.sqlalchemy_insert_one(qry_log, return_field = None)

        logger.info(f"RAG Query log data saved successfully.")

    def getQuery(self, ragReqDataModel: RagReqDataModel):
        """
        Extracts the latest user query from the RAG request model.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input object containing user inputs.

        Returns:
            str: The first available textual query from the input.
        """

        query = ""
        userInputs = ragReqDataModel.user_input
        print(userInputs)
        if userInputs:
            queryLst = [ui.content for ui in userInputs if (ui.type == "text" or ui.type == "audio") and ui.content]
            query = queryLst[0]

        return query

    def getChatHistory(self, ragReqDataModel: RagReqDataModel) -> List[Dict[str, str]]:
        """
        Constructs the chat history as a list of query-response pairs for use in the prompt.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input containing past dialog history.

        Returns:
            List[ChatHistory]: List of ChatHistory objects representing prior exchanges.
        """

        dialogDetails = ragReqDataModel.context.dialogDetails
        chatHistory: List[ChatHistory] = []
        for dialog in dialogDetails:
            userInputs: List[UserInput] = dialog.user_input
            if userInputs:
                queryLst = [ui.content for ui in userInputs if (ui.type == "text" or ui.type == "audio") and ui.content]
                queryStr = queryLst[0]

            chatHistory.append(
                ChatHistory(query= queryStr, response= dialog.response.text)
            )
        print(chatHistory)
        return chatHistory


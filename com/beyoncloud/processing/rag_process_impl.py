import time
import logging
from typing import Dict, List, Any
from datetime import datetime
from langchain.schema import Document
from com.beyoncloud.processing.generation.generator import RagGeneratorProcess
from com.beyoncloud.schemas.rag_reqres_data_model import (
    RagReqDataModel, RagRespDataModel,
    ChatHistory,UserInput,ResponseData, 
    SearchResult,StructureInputData, 
    EntityResponse, RagLogQryModel
)
from com.beyoncloud.db.postgresql_impl import PostgresSqlImpl
from com.beyoncloud.db.postgresql_connectivity import PostgreSqlConnectivity
from com.beyoncloud.config.settings.table_mapper_config import TableSettings
from com.beyoncloud.utils.file_util import TextLoader
import com.beyoncloud.config.settings.env_config as config

logger = logging.getLogger(__name__)

class RagProcessImpl:
    """
    Implementation class for the end-to-end Retrieval-Augmented Generation (RAG) process.

    This class handles the integration between retrieval and generation components.
    It supports both standard and LangChain-based RAG workflows and manages input parsing,
    chat history construction, and model orchestration.

    Author: Jenson
    Date: 16-June-2025
    """

    def __init__(self):
        """
        Initializes an instance of RagProcessImpl.
        """
        self.table_settings = TableSettings()

    async def generateRAGResponse(self, ragReqDataModel: RagReqDataModel, search_result: list[dict[str, Any]] = []):
        """
        Executes the basic RAG flow using a semantic search and a query string.

        Args:
            RagReqDataModel: Request input data model

        Returns:
            str: The generated answer based on retrieved context.
        """

        starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()
        query = self.getQuery(ragReqDataModel)
        orgId = self.getOrgId(ragReqDataModel)

        ragGeneratorProcess = RagGeneratorProcess()
        chatHistory = self.getChatHistory(ragReqDataModel)
        response = await ragGeneratorProcess.generateAnswer(ragReqDataModel,query, search_result, chatHistory)
        endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        end_time = time.time()
        elapsed = end_time - start_time  # in seconds (float)
        print(f"Start time : {starttime} --> End time : {endtime} --> elapsed: {elapsed}")
        #print(response)
        try:
            metadata = {
                "domain": ragReqDataModel.domain_id,
                "starttime": starttime,
                "start_time": start_time,
                "endtime": endtime,
                "end_time": end_time,
                "elapsed": elapsed
            }
            search_result_json = []
            if search_result:
                search_result_json = [
                    {
                        'id': result.get('id'),
                        'chunk_id': result.get('chunk_id'),
                        'chunk': result.get('chunk'),
                        'entities': result.get('entities'),
                        'matched_entities': result.get('matched_entities'),
                        'entity_match': result.get('entity_match'),
                        'entity_match_score': result.get('entity_match_score'),
                        'similarity_output': result.get('similarity_output')
                    }
                    for result in search_result
                ]

            rag_log_qry_model = RagLogQryModel(
                orgId = orgId,
                query = query,
                response = response,
                search_result_json = search_result_json,
                time_elapsed = elapsed,
                metadata = metadata
            )
            await self.saveRagResponse(rag_log_qry_model)
        except Exception as e:
            logger.error(f"Error processing RAG query log save : {e}")

        ragRespDataModel = self.getRagRespDataModel(ragReqDataModel, response)
        return ragRespDataModel

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

    def getOrgId(self, ragReqDataModel: RagReqDataModel) -> int:
        """
        Extracts the organization ID from the RAG request.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input object containing metadata.

        Returns:
            int: The organization ID.
        """

        orgId = ragReqDataModel.org_id
        return orgId

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
                queryLst = [ui.content for ui in userInputs if ui.type == "text" and ui.content]
                queryStr = queryLst[0]

            chatHistory.append(
                ChatHistory(query= queryStr, response= dialog.response.text)
            )
        print(chatHistory)
        return chatHistory

    def getInfRespDataModel(self, response: str) -> ResponseData:
        """
        Constructs the response data model from the RAG response.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input containing past dialog history.
            response: str: LLM returning the response string

        Returns:
            RagRespDataModel: Response data model for the requested 'RagReqDataModel' input.
        """

        responseData = ResponseData(
            text = response
        )

        return responseData

    async def generate_structured_response(self, structure_input_data: StructureInputData) -> str:
        """
        Executes the basic RAG flow using a semantic search and a query string.

        Args:
            RagReqDataModel: Request input data model

        Returns:
            str: The generated answer based on retrieved context.
        """

        starttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()
        rag_generator_process = RagGeneratorProcess()
        if config.ENABLE_HF_INFRENCE_YN == "Y":
            response = await rag_generator_process.hf_response(structure_input_data)
        else:
            response = await rag_generator_process.generate_structured_response(structure_input_data)

        
        endtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        end_time = time.time()
        elapsed = end_time - start_time  # in seconds (float)
        print(f"Start time : {starttime} --> End time : {endtime} --> elapsed: {elapsed}")
        #print(response)
        try:
            
            fullContext = structure_input_data.context_data
            file_path = structure_input_data.source_path
            #if structure_input_data.source_path:
            #    text_loader = TextLoader()
            #    fullContext = text_loader.get_text_content(structure_input_data.source_path)
            #    file_path = structure_input_data.source_path
            search_result_json = [fullContext]

            metadata = {
                "starttime": starttime,
                "start_time": start_time,
                "endtime": endtime,
                "end_time": end_time,
                "elapsed": elapsed,
                "file_path": file_path
            }

            rag_log_qry_model = RagLogQryModel(
                orgId = 0,
                query = "",
                response = response,
                search_result_json = search_result_json,
                time_elapsed = elapsed,
                metadata = metadata
            )
            await self.saveRagResponse(rag_log_qry_model)

        except Exception as e:
            logger.error(f"Error processing RAG query log save : {e}")

        #rag_resp_datamodel = self.get_entity_resp_model(structure_input_data, response)
        return response

    def get_entity_resp_model(self, structure_input_data: StructureInputData, response: str) -> EntityResponse:
        """
        Constructs the response data model from the RAG response.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input containing past dialog history.
            response: str: LLM returning the response string

        Returns:
            RagRespDataModel: Response data model for the requested 'RagReqDataModel' input.
        """

        response_model = EntityResponse(
            response = response,
            status = "Success"
        )

        return response_model

    async def saveRagResponse(
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

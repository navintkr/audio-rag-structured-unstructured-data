import re
from typing import Any
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection
import os
from dotenv import load_dotenv
from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import config

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def _search_tool(search_client: SearchClient, args: Any) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    print(f"Selected '{config.selected_option}' Data Type.")
    # Hybrid + Reranking query using Azure AI Search
    if config.selected_option=="UnstructuredData":
        search_results = await search_client.search(
            search_text=args['query'], 
            query_type="semantic",
            top=5,
            vector_queries=[VectorizableTextQuery(text=args['query'], k_nearest_neighbors=50, fields="text_vector")],
            select="chunk_id,title,chunk")
        result = ""
        async for r in search_results:
            result += f"[{r['chunk_id']}]: {r['chunk']}\n-----\n"
        return ToolResult(result, ToolResultDirection.TO_SERVER)
    
    else:
        #Using Lanchain to convert Text to SQL
        driver = '{ODBC Driver 17 for SQL Server}'
        odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                        'Driver='+driver+ \
                        ';Server=tcp:' + os.getenv("AZURE_SQL_SERVER")+'.database.windows.net;PORT=1433' + \
                        ';DATABASE=' + os.getenv("AZURE_SQL_DB") + \
                        ';Uid=' + os.getenv("AZURE_SQL_USERNAME")+ \
                        ';Pwd=' + os.getenv("AZURE_SQL_PWD") + \
                        ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        db_engine = create_engine(odbc_str)
        llm = AzureChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"),
                            deployment_name=os.getenv("OPENAI_CHAT_MODEL"),
                            temperature=0)
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                "You are an AI assistant tasked with generating SQL queries from natural language requests." +\
                "Use only the schema provided and do not reference any other tables. Always refer to the following table Schemas: " +\
                "CREATE TABLE [dbo].[Customers]([CustomerID] [int] NOT NULL,[CustomerName] [varchar](255) NULL,[Email] [varchar](255) NULL) ON [PRIMARY]GO ALTER TABLE [dbo].[Customers] ADD PRIMARY KEY CLUSTERED ([CustomerID] ASC)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY] GO. " +\
                "CREATE TABLE [dbo].[Orders]([OrderID] [int] NOT NULL,[CustomerID] [int] NULL,[OrderDate] [date] NULL,[TotalAmount] [decimal](10, 2) NULL) ON [PRIMARY] GO ALTER TABLE [dbo].[Orders] ADD PRIMARY KEY CLUSTERED ([OrderID] ASC) WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY] GO ALTER TABLE [dbo].[Orders]  WITH CHECK ADD FOREIGN KEY([CustomerID]) REFERENCES [dbo].[Customers] ([CustomerID]) GO. " +\
                "Use this schema to generate accurate SQL queries based on the natural language input provided. " +\
                "If the request cannot be fulfilled with the available data, respond with - We don’t have data available for this request, " +\
                "please try something else! but don't show all internal attempts to generate the response." +\
                "Always make sure the query results not more than 100 best rows so that I wont exceed context length when passing to LLM again. " +\
                "Do not add ``` or backticks to the sql query that your are generating. The query is for Azure SQL so dont add LIMIT instead use TOP or OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY. Example Queries: " +\
                "Query: Give the list of customers who ordered at least once. SQL:SELECT TOP 10 CustomerName FROM dbo.Customers WHERE CustomerID IN (SELECT DISTINCT CustomerID FROM dbo.Orders) ORDER BY CustomerID; " +\
                "Query: Give me the list of customers who ordered twice. SQL:SELECT C.CustomerID, C.CustomerName, C.Email FROM dbo.Customers C JOIN dbo.Orders O ON C.CustomerID = O.CustomerID GROUP BY C.CustomerID, C.CustomerName, C.Email HAVING COUNT(O.OrderID) = 2;" +\
                "Query: How can I retrieve the names and emails of all customers who have placed an order? SQL: SELECT DISTINCT c.CustomerName, c.Email FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID; "
                
                ),
                ("user", "{question}\n ai: "),
            ]
        )
        db = SQLDatabase(db_engine)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_toolkit.get_tools()
        sqldb_agent = create_sql_agent(
            llm=llm,
            toolkit=sql_toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        result = sqldb_agent.run(final_prompt.format(question=args['query']))
        print(result)
        return ToolResult(result, ToolResultDirection.TO_SERVER)


KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# TODO: move from sending all chunks used for grounding eagerly to only sending links to 
# the original content in storage, it'll be more efficient overall
async def _report_grounding_tool(search_client: SearchClient, args: Any) -> None:
    sources = [s for s in args["sources"] if KEY_PATTERN.match(s)]
    list = " OR ".join(sources)
    print(f"Grounding source: {list}")
    # Use search instead of filter to align with how detailt integrated vectorization indexes
    # are generated, where chunk_id is searchable with a keyword tokenizer, not filterable 
    search_results = await search_client.search(search_text=list, 
                                                search_fields=["chunk_id"], 
                                                select=["chunk_id", "title", "chunk"], 
                                                top=len(sources), 
                                                query_type="full")
    
    # If your index has a key field that's filterable but not searchable and with the keyword analyzer, you can 
    # use a filter instead (and you can remove the regex check above, just ensure you escape single quotes)
    # search_results = await search_client.search(filter=f"search.in(chunk_id, '{list}')", select=["chunk_id", "title", "chunk"])

    docs = []
    async for r in search_results:
        docs.append({"chunk_id": r['chunk_id'], "title": r["title"], "chunk": r['chunk']})
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier, search_endpoint: str, search_index: str, credentials: AzureKeyCredential | DefaultAzureCredential) -> None:
    if not isinstance(credentials, AzureKeyCredential):
        credentials.get_token("https://search.azure.com/.default") # warm this up before we start getting requests
    search_client = SearchClient(search_endpoint, search_index, credentials, user_agent="RTMiddleTier")

    rtmt.tools["search"] = Tool(schema=_search_tool_schema, target=lambda args: _search_tool(search_client, args))
    rtmt.tools["report_grounding"] = Tool(schema=_grounding_tool_schema, target=lambda args: _report_grounding_tool(search_client, args))

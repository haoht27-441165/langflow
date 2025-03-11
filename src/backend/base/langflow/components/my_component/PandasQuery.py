import importlib

from langchain.tools import StructuredTool
from langchain_core.tools import ToolException

from langchain_experimental.utilities import PythonREPL
import pandas as pd
from io import StringIO

from loguru import logger
from pydantic import BaseModel, Field

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import StrInput, DataInput
from langflow.schema import Data



class PandasToolComponent(LCToolComponent):
    display_name = "Python Pandas"
    description = "A tool for query Dataframe use Pandas."
    name = "PandasTool"
    icon = "Python"

    inputs = [
        DataInput(name="data", display_name="upload file csv", info="data"),
        
        # StrInput(name="path", display_name="path file csv", info="something"),
        StrInput(name="query", display_name="Query", info="The query to run on the DataFrame."),
    ]

    class PandasToolSchema(BaseModel):

        query: str = Field(
            ...,
            title="Query",
            description="The query to run on the DataFrame.",
        )


    def build_tool(self) -> Tool:
        
        def _run_query(query:str) -> str:
            try:
                # Use eval to execute the pandas query
                # df = pd.read_csv(path)
                text_data = StringIO(self.data.get_text())
                df = pd.read_csv(text_data)
                
                result = df.query(query)
                return result.to_string(index=False)
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        tool = StructuredTool.from_function(
            name=self.name,
            description=self.description,
            func=_run_query,
            args_schema=self.PandasToolSchema,
        )
        return tool

   
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        # result = tool.run(self.query)
        return self._run_query(self.query)


   

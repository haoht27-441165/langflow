import importlib

from langchain.tools import StructuredTool
from langchain_core.tools import ToolException

from langchain_experimental.utilities import PythonREPL
import pandas as pd

from loguru import logger
from pydantic import BaseModel, Field

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import StrInput, DataInput
from langflow.schema import Data



class SaveTextComponent(LCToolComponent):
    display_name = "Python text save"
    description = "A tool for text save."
    name = "text_save"
    icon = "Python"

    inputs = [
        StrInput(name="text", display_name="text to save", info="something"),

    ]

    class SaveTextSchema(BaseModel):
        text: str = Field(
            ...,
            title="text",
            description="text save",
        )


    def build_tool(self) -> Tool:
        
        def _run_save_text(text:str) -> str:
            try:
                # Use eval to execute the pandas query
                with open('/home/hadoop/Agent_code/lf-project/test_save.txt', 'w') as f:
                    f.write(text) 
                # while
                # df = pd.read_csv(self.path)
                
                # result = df.query(query)
                return 'Done save text'
            
            except Exception as e:
                return f"Error executing saving: {str(e)}"
        
        tool = StructuredTool.from_function(
            name=self.name,
            description=self.description,
            func=_run_save_text,
            args_schema=self.SaveTextSchema,
        )
        return tool

   
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        # result = tool.run(self.query)
        return self._run_save_text(self.text)


   

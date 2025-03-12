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
    description = "A test tool for save file txt, can use PythonREPL don't use me."
    name = "text_save"
    icon = "Python"

    inputs = [
        StrInput(name="text", display_name="text to save", info="something"),
        StrInput(name="path", display_name="path to save", info="something"),

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
                
                if self.path:
                   if not self.path.endswith(".txt"):
                       
                        old_path = self.path.split('/')
                        if '.' not in old_path[-1]:
                            file_name = 'new_file.txt'
                            old_path.append(file_name)
                        else:
                            file_name = old_path[-1].split('.')
                            file_name = file_name[0]+'.txt'
                            old_path[-1] = file_name
              
                        self.path = '/'.join(old_path)                # save
                        with open(self.path, 'w') as f:
                            f.write(text) 
          
                        return 'Done save text'
                else :
                    raise ValueError("Path is empty. Cannot save file.")

                
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


   

import operator
from functools import reduce

from langchain_community.llms import VLLM
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI

from pydantic.v1 import SecretStr

import httpx
from urllib.parse import urljoin

from langflow.field_typing.range_spec import RangeSpec
from langflow.field_typing import LanguageModel
from langflow.base.models.model import LCModelComponent
from langflow.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    IntInput,
    SecretStrInput,
    StrInput,
)


class VLLMComponent(LCModelComponent):
    display_name = "vllm"
    description = "Generate text using vllm"
    icon = "🖕"
    name = "VLLM"

    inputs = [
        StrInput(
            name="base_url",
            display_name="Base URL",
            info="Endpoint of the VLLM API. Defaults to 'http://localhost:8000/v1' if not specified.",
            value="http://localhost:8000",
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=None,
            advanced=True,
            info="The maximum number of tokens to generate. Set to 1 for unlimited tokens.",
            range_spec=RangeSpec(min=1, max=128000),
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            value="Qwen/Qwen2.5-0.5B-Instruct",
            info="Refer to me for more models.",
            refresh_button=True,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(step_type='float',min=0.0, max=1.0),
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            value=0.95,
            range_spec=RangeSpec(step_type='float',min=0.0, max=1.0),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        *LCModelComponent._base_inputs,
    ]

    def update_build_config(
        self, build_config: dict, field_value: str, field_name: str | None = None
    ):
        # build_config[a]["value"] = field_value

        if field_name == "model_name":
            base_url_dict = build_config.get("base_url", {})
            # base_url_load_from_db = base_url_dict.get("load_from_db", False)
            base_url_value = base_url_dict.get("value")
            if not base_url_dict:
                base_url_value = "http://localhost:8000"
            build_config["model_name"]["options"] = self.get_model(base_url_value)

        return build_config

    def get_model(self, base_url_value: str) -> list[str]:
        try:
            url = urljoin(base_url_value, "/v1/models")
            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

                return [data["data"][0]["id"]]
        except Exception as e:
            msg = "Could not retrieve models. Please, make sure Vllm is running."
            raise ValueError(msg) from e

    def build_model(self) -> LanguageModel :
        # print(urljoin(self.base_url, "/v1"))
        
        # output = VLLMOpenAI(
        #     openai_api_key="EMPTY",
        #     openai_api_base=urljoin(self.base_url, "/v1"),
        #     model_name=self.model_name,        
        #     max_tokens=self.max_tokens,
        #     # max_tokens=None,
        #     top_p=self.top_p,
        #     temperature=self.temperature,
        #     model_kwargs=self.model_kwargs or {}
        # )

        output = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=urljoin(self.base_url, "/v1"),
            model_name=self.model_name,        
            max_tokens=self.max_tokens,
            # max_tokens=None,
            top_p=self.top_p,
            temperature=self.temperature,
            model_kwargs=self.model_kwargs or {}
        )
        
        return output
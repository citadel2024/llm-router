import logging
from typing import Any

import pytest

from src.config import LogConfiguration
from src.config.config import LLMProviderConfig, RouterConfig, LoadBalancerConfig
from src.providers.base_provider import BaseLLMProvider


def test_default_values():
    config = LogConfiguration()
    assert config.stage == "dev"
    assert config.level == logging.DEBUG
    assert config.log_dir == "logs"


def test_custom_values():
    config = LogConfiguration(stage="prod", level=logging.INFO, log_dir="custom_logs")
    assert config.stage == "prod"
    assert config.level == logging.INFO
    assert config.log_dir == "custom_logs"


@pytest.mark.parametrize("stage", ["dev", "prod"])
def test_stage_param(stage):
    config = LogConfiguration(stage=stage)
    assert config.stage == stage


def test_invalid_stage():
    with pytest.raises(ValueError):
        LogConfiguration(stage="invalid")


class MockLLMProvider(BaseLLMProvider):
    async def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> Any:
        pass

    async def embedding(self, texts: list[str], **kwargs) -> Any:
        pass


def test_provider_config_type_error():
    gpt3_impl = MockLLMProvider()
    with pytest.raises(TypeError):
        LLMProviderConfig(impl=gpt3_impl, rpm=100, tpm=100)


def test_provider_unique_id():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=100, tpm=100)

    assert gpt3.id == "f8d1b37b8f2cc0a6cdc71b3e646ec685b9b45b407cd987efbe06d0536a81b02d"


def test_failed_init_provider_with_id():
    gpt3_impl = MockLLMProvider()
    with pytest.raises(TypeError, match="LLMProviderConfig\.__init__\(\) got an unexpected keyword argument 'id'"):
        LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=-1, tpm=100, id=1)


def test_validate_integer():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=100, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_id="llama", impl=llama_impl, rpm=100, tpm=100)
    with pytest.raises(ValueError):
        RouterConfig(
            llm_provider_group={"gpt3-level-model": [gpt3, llama]},
            cooldown_seconds=-1,
        )
    with pytest.raises(ValueError):
        RouterConfig(
            llm_provider_group={"gpt3-level-model": [gpt3, llama]},
            num_retries=-1,
        )
    with pytest.raises(ValueError):
        RouterConfig(
            llm_provider_group={"gpt3-level-model": [gpt3, llama]},
            timeout_seconds=-1,
        )


def test_validate_invalid_capacity_dimension():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=100, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_id="llama", impl=llama_impl, rpm=100, tpm=100)
    with pytest.raises(ValueError, match="Invalid capacity dimension: invalid"):
        RouterConfig(
            llm_provider_group={"gpt3-level-model": [gpt3, llama]},
            load_balancer_config=LoadBalancerConfig(capacity_dimension="invalid"),
        )


def test_validate_capacity_dimension_missing_value():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_id="llama", impl=llama_impl, tpm=100)
    dimension = "rpm"
    with pytest.raises(ValueError, match=f"Capacity dimension {dimension} is not found."):
        RouterConfig(
            llm_provider_group={"gpt3-level-model": [gpt3, llama]},
            load_balancer_config=LoadBalancerConfig(capacity_dimension=dimension),
        )


def test_create_route_config():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=100, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_id="llama", impl=llama_impl, rpm=100, tpm=100)
    rc = RouterConfig(
        llm_provider_group={"gpt3-level-model": [gpt3, llama]},
        load_balancer_config=LoadBalancerConfig(capacity_dimension="rpm"),
    )

    assert len(rc.llm_provider_group) == 1
    assert len(rc.llm_provider_group["gpt3-level-model"]) == 2
    assert rc.llm_provider_group["gpt3-level-model"][0].model_id == "gpt3"
    assert rc.llm_provider_group["gpt3-level-model"][0].impl == gpt3_impl
    assert rc.llm_provider_group["gpt3-level-model"][0].rpm == 100
    assert rc.llm_provider_group["gpt3-level-model"][0].tpm == 100

    assert rc.llm_provider_group["gpt3-level-model"][1].model_id == "llama"
    assert rc.llm_provider_group["gpt3-level-model"][1].impl == llama_impl
    assert rc.llm_provider_group["gpt3-level-model"][1].rpm == 100
    assert rc.llm_provider_group["gpt3-level-model"][1].tpm == 100


# skip
@pytest.mark.skip
def test_to_json():
    gpt3_impl = MockLLMProvider()
    gpt3 = LLMProviderConfig(model_id="gpt3", impl=gpt3_impl, rpm=100, tpm=100)
    llama_impl = MockLLMProvider()
    llama = LLMProviderConfig(model_id="llama", impl=llama_impl, rpm=100, tpm=100)
    rc = RouterConfig(
        llm_provider_group={"gpt3-level-model": [gpt3, llama]},
    )
    actual = rc.to_json()
    # print(rc.to_json(2))
    expect = (
        '{'
        '  "cooldown_seconds":60,'
        '  "lb_strategy": "LoadBalanceStrategy.CAPACITY_BASED_BALANCING",'

        '  "llm_provider_group": {'
        '    "gpt3-level-model": ['
        '      {'
        '        "id":"f8d1b37b8f2cc0a6cdc71b3e646ec685b9b45b407cd987efbe06d0536a81b02d",'
        '        "impl": "<tests.config.test_config.MockLLMProvider>", '
        '        "model_id": "gpt3", '
        '        "rpm": 100, '
        '        "tpm": 100'
        '      }, '
        '      {'
        '        "id":"08a9167a804d07b07ad515a901a58da2b4d64a890e9db347aac981c28dfe17bc",'
        '        "impl": "<tests.config.test_config.MockLLMProvider>", '
        '        "model_id": "llama", '
        '        "rpm": 100, '
        '        "tpm": 100'
        '      }'
        '    ]'
        '  }, '
        '  "log_config": {'
        '    "level": 10,'
        '    "log_dir": "logs",'
        '    "stage": "dev"'
        '  },'
        '  "num_retries":3,'
        '  "retry_policy":null,'
        '  "timeout_seconds":30'
        '}'
    )
    assert actual == expect.replace(" ", "")

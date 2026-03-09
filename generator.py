import time
import json
import openai
import logging
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .prompts import PromptTemplates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM生成器类，包含完整的重试策略"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 300,
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 api_key: Optional[str] = None):

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

        if api_key:
            openai.api_key = api_key

        self.cache = {}

    def _should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        retryable_exceptions = (
            openai.error.RateLimitError,
            openai.error.APIConnectionError,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
            openai.error.APIError
        )
        return isinstance(exception, retryable_exceptions)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((
                openai.error.RateLimitError,
                openai.error.APIConnectionError,
                openai.error.Timeout
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """调用OpenAI API（带重试机制）"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            return response.choices[0].message.content

        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid request error: {e}")
            raise
        except openai.error.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise
        except Exception as e:
            if self._should_retry(e):
                logger.warning(f"Retryable error occurred: {e}")
                raise
            else:
                logger.error(f"Non-retryable error: {e}")
                raise

    def generate_image_description(self,
                                   age: str,
                                   education: str,
                                   work: str,
                                   attention: str,
                                   visual_features: str,
                                   ocr_text: str,
                                   news_excerpt: str,
                                   use_cache: bool = True) -> str:
        """生成多视角图像描述"""
        cache_key = f"nr_{age}_{education}_{work}_{attention}_{hash(visual_features)}"

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        messages = [
            {"role": "system", "content": PromptTemplates.NRP_SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.NRP_USER_TEMPLATE.format(
                age=age,
                education=education,
                work=work,
                attention=attention,
                visual_features_summary=visual_features[:200],
                ocr_text=ocr_text[:200],
                news_document_excerpt=news_excerpt[:200]
            )}
        ]

        try:
            response = self._call_openai_api(messages)
            words = response.split()
            if len(words) > 100:
                response = ' '.join(words[:100])

            if use_cache:
                self.cache[cache_key] = response

            return response

        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return ""

    def generate_external_entities(self,
                                   full_news_document: str,
                                   initial_entities: List[str],
                                   use_cache: bool = True) -> Dict[str, Any]:
        """生成外部实体"""
        cache_key = f"eesp_{hash(full_news_document)}_{hash(str(initial_entities))}"

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        entities_str = "\n".join([f"- {entity}" for entity in initial_entities])

        messages = [
            {"role": "system", "content": PromptTemplates.EESP_SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.EESP_USER_TEMPLATE.format(
                full_news_document=full_news_document[:2000],
                initial_entities=entities_str
            )}
        ]

        try:
            response = self._call_openai_api(messages)

            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed, returning empty result")
                result = {"newly_added_entities": [], "implicit_entities": [], "initial_entities_with_importance": []}

            if use_cache:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to generate entities: {e}")
            return {"newly_added_entities": [], "implicit_entities": [], "initial_entities_with_importance": []}
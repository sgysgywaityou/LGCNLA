"""
LLM生成器模块 - 包含指数退避重试策略和完整的生成逻辑
"""

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
import openai.error

from .prompts import PromptTemplates

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM生成器类，包含完整的重试策略和错误处理"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0613",
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 300,
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 api_key: Optional[str] = None):
        """
        初始化LLM生成器

        Args:
            model_name: OpenAI模型名称
            temperature: 温度参数（设为0以保证确定性）
            top_p: top_p采样参数
            max_tokens: 最大生成token数
            max_retries: 最大重试次数
            base_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
            api_key: OpenAI API密钥
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

        if api_key:
            openai.api_key = api_key

        # 用于缓存生成结果（提高可复现性）
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
        """
        调用OpenAI API（带重试机制）

        Args:
            messages: 对话消息列表

        Returns:
            生成的文本内容
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                seed=42  # 固定种子以提高确定性
            )
            return response.choices[0].message.content

        except openai.error.InvalidRequestError as e:
            # 永久错误，不重试
            logger.error(f"Invalid request error: {e}")
            raise
        except openai.error.AuthenticationError as e:
            # 认证错误，不重试
            logger.error(f"Authentication error: {e}")
            raise
        except Exception as e:
            # 其他错误，根据策略决定是否重试
            if self._should_retry(e):
                logger.warning(f"Retryable error occurred: {e}")
                raise  # 触发重试
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
        """
        生成多视角图像描述（NRP）

        Args:
            age: 年龄属性
            education: 教育属性
            work: 工作属性
            attention: 关注度属性
            visual_features: 视觉特征摘要
            ocr_text: OCR提取的文本
            news_excerpt: 新闻文档摘要
            use_cache: 是否使用缓存

        Returns:
            生成的图像描述
        """
        # 构建缓存键
        cache_key = f"nr_{age}_{education}_{work}_{attention}_{hash(visual_features)}_{hash(ocr_text)}"

        if use_cache and cache_key in self.cache:
            logger.info(f"Using cached description for {cache_key}")
            return self.cache[cache_key]

        # 构建消息
        messages = [
            {"role": "system", "content": PromptTemplates.NRP_SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.NRP_USER_TEMPLATE.format(
                age=age,
                education=education,
                work=work,
                attention=attention,
                visual_features_summary=visual_features[:200],  # 限制长度
                ocr_text=ocr_text[:200],
                news_document_excerpt=news_excerpt[:200]
            )}
        ]

        try:
            response = self._call_openai_api(messages)

            # 后处理：长度规范化
            words = response.split()
            if len(words) > 100:
                response = ' '.join(words[:100])

            # 缓存结果
            if use_cache:
                self.cache[cache_key] = response

            return response

        except Exception as e:
            logger.error(f"Failed to generate image description after retries: {e}")
            # 返回空字符串作为fallback
            return ""

    def generate_external_entities(self,
                                   full_news_document: str,
                                   initial_entities: List[str],
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        生成外部实体（EESP）

        Args:
            full_news_document: 完整新闻文档
            initial_entities: SpaCy提取的初始实体列表
            use_cache: 是否使用缓存

        Returns:
            包含新增实体和隐式实体的字典
        """
        # 构建缓存键
        cache_key = f"eesp_{hash(full_news_document)}_{hash(str(initial_entities))}"

        if use_cache and cache_key in self.cache:
            logger.info(f"Using cached entities for {cache_key}")
            return self.cache[cache_key]

        # 格式化初始实体
        entities_str = "\n".join([f"- {entity}" for entity in initial_entities])

        # 构建消息
        messages = [
            {"role": "system", "content": PromptTemplates.EESP_SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.EESP_USER_TEMPLATE.format(
                full_news_document=full_news_document[:2000],  # 限制长度
                initial_entities=entities_str
            )}
        ]

        try:
            response = self._call_openai_api(messages)

            # 解析JSON响应
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # 尝试修复JSON
                logger.warning("JSON parsing failed, attempting repair...")
                result = self._repair_json_response(response)

            # 验证schema
            if not self._validate_entity_schema(result):
                logger.error("Invalid entity schema, retrying once...")
                # 重试一次
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": "The output format was invalid. Please ensure valid JSON with the specified schema."})
                response = self._call_openai_api(messages)
                try:
                    result = json.loads(response)
                except:
                    result = {"newly_added_entities": [], "implicit_entities": [],
                              "initial_entities_with_importance": []}

            # 缓存结果
            if use_cache:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to generate external entities after retries: {e}")
            return {"newly_added_entities": [], "implicit_entities": [], "initial_entities_with_importance": []}

    def _repair_json_response(self, response: str) -> Dict[str, Any]:
        """修复损坏的JSON响应"""
        import re

        result = {
            "newly_added_entities": [],
            "implicit_entities": [],
            "initial_entities_with_importance": []
        }

        # 尝试用正则提取实体
        patterns = [
            (r'"newly_added_entities":\s*(\[.*?\])', "newly_added_entities"),
            (r'"implicit_entities":\s*(\[.*?\])', "implicit_entities"),
            (r'"initial_entities_with_importance":\s*(\[.*?\])', "initial_entities_with_importance")
        ]

        for pattern, key in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    result[key] = json.loads(match.group(1))
                except:
                    pass

        return result

    def _validate_entity_schema(self, result: Dict[str, Any]) -> bool:
        """验证实体schema"""
        required_keys = ["newly_added_entities", "implicit_entities", "initial_entities_with_importance"]

        if not all(key in result for key in required_keys):
            return False

        for key in required_keys:
            if not isinstance(result[key], list):
                return False

        return True

    def save_cache(self, filepath: str):
        """保存缓存到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.cache, f)

    def load_cache(self, filepath: str):
        """从文件加载缓存"""
        try:
            with open(filepath, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Cache file {filepath} not found")
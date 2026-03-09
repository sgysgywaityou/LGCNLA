class PromptTemplates:
    """提示词模板类"""

    # ==================== NRP: News Readers Prompt ====================
    NRP_SYSTEM_PROMPT = """You are a news reader with specific demographic attributes. Based on these attributes, you will provide a concise description of a news image. Your description should:
1. Focus on the factual content visible in the image
2. Reflect how your background might influence your interpretation
3. Be written in a news narrative style
4. Be concise (maximum 100 words)
5. Avoid speculation beyond what can be reasonably inferred from the image"""

    NRP_USER_TEMPLATE = """Your attributes:
- Age: {age}
- Education: {education}
- Work: {work}
- News attention level: {attention}

Image Information:
- Visual description: [The image contains: {visual_features_summary}]
- Text detected in image (OCR): {ocr_text}

News context (for reference only, do not repeat): {news_document_excerpt}

Please provide a concise description of this image from your perspective as this specific reader."""

    # 属性值映射
    AGE_VALUES = ["18-35", "36-55", ">55"]
    EDUCATION_VALUES = ["High school or below", "Bachelor's degree", "Master's degree", "Doctoral degree"]
    WORK_VALUES = ["Related to news", "Not related to news"]
    ATTENTION_VALUES = ["Less than 2 hours per week", "2-8 hours per week", "More than 8 hours per week"]

    # ==================== EESP: External Entity Supplementary Prompt ====================
    EESP_SYSTEM_PROMPT = """You are a senior domain expert in news verification and fact-checking. Your task is to analyze a news document and its initial extracted entities, then identify additional relevant entities that could help verify the news authenticity. You should distinguish between explicit entities (directly mentioned) and implicit entities (implied or contextually relevant)."""

    EESP_USER_TEMPLATE = """News Document:
{full_news_document}

Initial External Entities (extracted by SpaCy):
{initial_entities}

Please complete the following tasks:
1. Identify NEWLY ADDED entities that appear in the news document but are missing from the initial extraction
2. Identify IMPLICIT entities that are not directly mentioned but are contextually relevant for fact-checking
3. Classify the importance of ALL entities (including initial ones) into three levels: HIGH, MEDIUM, LOW

Output Format (JSON):
{{
  "newly_added_entities": [
    {{"name": "entity_name", "importance": "HIGH/MEDIUM/LOW", "reasoning": "brief explanation"}}
  ],
  "implicit_entities": [
    {{"name": "entity_name", "importance": "HIGH/MEDIUM/LOW", "reasoning": "brief explanation"}}
  ],
  "initial_entities_with_importance": [
    {{"name": "initial_entity_name", "importance": "HIGH/MEDIUM/LOW"}}
  ]
}}

Important: Only include entities with HIGH or MEDIUM importance in the final output. LOW importance entities should be omitted."""
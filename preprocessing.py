import re
import jieba  # 用于中文分词
import spacy

# 加载语言模型
nlp_en = spacy.load('en_core_web_sm')
nlp_zh = spacy.load('zh_core_web_sm')


def preprocess_text(text, language='en'):
    """文本预处理"""
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 去除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 去除特殊字符
    text = re.sub(r'[^\w\s]', '', text)

    # 转换为小写（仅英文）
    if language == 'en':
        text = text.lower()

    return text.strip()


def extract_entities_spacy(text, language='en'):
    """使用SpaCy提取命名实体"""
    if language == 'en':
        doc = nlp_en(text)
    else:
        doc = nlp_zh(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })

    return entities


def tokenize_chinese(text):
    """中文分词"""
    return list(jieba.cut(text))
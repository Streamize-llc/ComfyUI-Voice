from . import korean
from . import cleaned_text_to_sequence
import copy

language_module_map = {"KR": korean}

def clean_text(text, language):
    m = language_module_map[language]
    norm_text = m.text_normalize(text)
    phones, tones, word2ph = m.g2p(norm_text)
    return norm_text, phones, tones, word2ph

def clean_text_bert(text, language, device=None):
    m = language_module_map[language]
    norm_text = m.text_normalize(text)
    phones, tones, word2ph = m.g2p(norm_text)
    w2 = copy.deepcopy(word2ph)
    for i in range(len(word2ph)): word2ph[i]*=2
    word2ph[0]+=1
    bert = m.get_bert_feature(norm_text, word2ph, device=device)
    return norm_text, phones, tones, w2, bert

def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

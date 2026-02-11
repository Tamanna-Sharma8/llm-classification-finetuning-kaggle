"""Feature engineering for LLM preference prediction
Author: Tamanna Sharma
Date: February 2026
"""

def extract_length_features(response_a, response_b):
    """Extract length-based features for LLM responses"""
    features = {
        'len_a': len(str(response_a)),
        'len_b': len(str(response_b)),
        'len_diff': len(str(response_a)) - len(str(response_b)),
        'len_ratio': len(str(response_a)) / (len(str(response_b)) + 1)  # avoid division by zero
    }
    return features

def extract_quality_features(response):
    """Extract quality indicators from a single LLM response"""
    text = str(response).lower()
    features = {
        'word_count': len(text.split()),
        'bullet_points': text.count('•') + text.count('-') + text.count('*'),
        'has_disclaimer': int('disclaimer' in text or 'note:' in text),
        'has_code': int('```' in text or 'def ' in text),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!')
    }
    return features

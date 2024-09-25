import re

def is_standard_english(text):
    # This regex pattern matches standard English characters, numbers, and basic punctuation
    pattern = r'^[a-zA-Z0-9\s.,!?()-_]+$'
    return bool(re.match(pattern, str(text))) 

def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def is_long_enough(text, length): 
    return len(str(text)) >= length

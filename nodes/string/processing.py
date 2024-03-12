import re

class SAIStringRegexSearchReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Text for replacement..."}),
                "regex_pattern": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "\\b\\w{5}\\b"}),
                "replacement_text": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Replacement text..."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("replaced_text",)

    FUNCTION = "replace_matches"
    CATEGORY = "SALT/String/Process/Regex"

    def replace_matches(self, text_input, regex_pattern, replacement_text):
        replaced_text = re.sub(regex_pattern, replacement_text, text_input)
        return (replaced_text,)


class SAIStringRegexSearchMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Text to search..."}),
                "regex_pattern": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "\\b[a-zA-Z]{6}\\b"}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("matches",)

    FUNCTION = "search_matches"
    CATEGORY = "SALT/String/Process/Regex"

    def search_matches(self, text_input, regex_pattern):
        matches = re.findall(regex_pattern, text_input)
        return (matches,)

NODE_CLASS_MAPPINGS = {
    "SAIStringRegexSearchReplace": SAIStringRegexSearchReplace,
    "SAIStringRegexSearchMatch": SAIStringRegexSearchMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAIStringRegexSearchReplace": "Regex Search and Replace",
    "SAIStringRegexSearchMatch": "Regex Search and Match"
}
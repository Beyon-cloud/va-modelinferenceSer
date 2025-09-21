import re

def extract_number(input_string):
    """
    Extracts the first number from a string that is enclosed by '```' symbols.

    Args:
        input_string (str): The string to process.

    Returns:
        str: The extracted number as a string, or None if no match is found.
    """
    delimiter = "```"
    # The regex pattern looks for a sequence of digits (\d+) that
    # is preceded and followed by the specified delimiter.
    match = re.search(fr'{delimiter}(\S+){delimiter}', input_string)
    if match:
        return match.group(1)
    return None

def extract_content(input_string):
    """
    Extracts the content from a multiline string that starts and ends with '```'.

    Args:
        input_string (str): The string to process.

    Returns:
        str: The extracted content as a string, or None if no match is found.
    """
    # The regex pattern looks for content (.*?) that is preceded by a '```'
    # and ends with another '```'. The '(?s)' flag allows the dot to match newlines.
    match = re.search(r'```(.*?)```', input_string, re.DOTALL)
    if match:
        return match.group(1)
    return None

# The input string provided
#input_data = '```123456```78910```111213```141516'
input_data= """```json
{
    "title": "Lease Agreement",
    "type": "object",
    "properties": {
        "lessor": {```
            "type": "object",
            "properties":``` {"""

# Extract the number
print(f"input_data --> {input_data}")
#extracted_value = extract_number(input_data)
extracted_value = extract_content(input_data)


# Print the result
if extracted_value:
    print(extracted_value)
else:
    print("No Match found.")

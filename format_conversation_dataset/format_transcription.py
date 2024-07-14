# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_format_transcription.ipynb.

# %% auto 0
__all__ = ['process_speakers', 'convert_file']

# %% ../nbs/00_format_transcription.ipynb 3
import json
import re

def process_speakers(input_text, assistant_speaker, system_context=None):
    """
    Process the input text and convert it to the desired JSON format.

    Args:
    input_text (str): The input text containing speaker dialogues.
    assistant_speaker (str): The speaker number to designate as the assistant.

    Returns:
    dict: A dictionary containing the processed messages.
    """
    output = [
        {
            "role": "system",
            "content": system_context if system_context else "You are participating in a conversation with one or more other speakers, and you are facilitating the conversation."
        },
    ]
    
    lines = input_text.strip().split('\n')
    current_user_content = []
    
    for line in lines:
        match = re.match(r'Speaker SPEAKER_(\d+):\s*(.*)', line)
        if match:
            speaker_num, content = match.groups()
            
            if speaker_num == assistant_speaker:
                if current_user_content:
                    output.append({
                        "role": "user",
                        "content": " ".join(current_user_content)
                    })
                    current_user_content = []
                
                output.append({
                    "role": "assistant",
                    "content": content.strip()
                })
            else:
                current_user_content.append(f"Speaker SPEAKER_{speaker_num}: {content.strip()}")
    
    if current_user_content:
        output.append({
            "role": "user",
            "content": " ".join(current_user_content)
        })
    
    return {"messages": output}

def convert_file(input_file_path, assistant_speaker, output_file_path=None, system_context=None):
    """
    Convert a file containing speaker dialogues to JSON format.

    Args:
    input_file_path (str): Path to the input text file.
    assistant_speaker (str): The speaker number to designate as the assistant.
    output_file_path (str, optional): Path to save the output JSON file. If not provided, returns the JSON string.

    Returns:
    str or None: If output_file_path is not provided, returns the JSON string. Otherwise, saves to file and returns None.
    """
    with open(input_file_path, 'r') as file:
        input_text = file.read()

    result = process_speakers(input_text, assistant_speaker, system_context)

    if output_file_path:
        with open(output_file_path, 'w') as file:
            json.dump(result, file, indent=4)
        return None
    else:
        return json.dumps(result, indent=4)

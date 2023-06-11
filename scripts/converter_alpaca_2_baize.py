import json
import re
import argparse

def fix_json_errors(json_string):
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    json_string = json_string.replace('\'', '\"')
    return json_string

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_string = json_file.read()

    try:
        return json.loads(json_string)
    except json.decoder.JSONDecodeError:
        fixed_json_string = fix_json_errors(json_string)
        try:
            return json.loads(fixed_json_string)
        except json.decoder.JSONDecodeError as e:
            print(f"Error while decoding JSON: {e}")
            return None

def convert_json(data):
    if data is None:
        return

    output_data = []

    for item in data:
        output_item = {}
        output_item['topic'] = item['instruction']

        if 'input' in item and item['input']:
            output_item['input'] = f"The conversation between human and AI assistant.\n[|Human|] {item['instruction']}\n{item['input']}\n[|AI|] {item['output']}\n[|Human|] "
        else:
            output_item['input'] = f"The conversation between human and AI assistant.\n[|Human|] {item['instruction']}\n[|AI|] {item['output']}\n[|Human|] "

        output_data.append(output_item)

    return json.dumps(output_data, separators=(',', ':'), indent=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files.')
    parser.add_argument('--file-input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--file-output', type=str, required=True, help='Path to output JSON file')

    args = parser.parse_args()
    input_data = load_json_file(args.file_input)
    conv_data = convert_json(input_data)

    with open(args.file_output, 'w') as output_file:
        output_file.write(conv_data)


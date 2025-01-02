import json


def extract_data(input_file, output_file):
    # Read the JSON data from the input file
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Extract the required fields
    extracted_data = []
    for item in data:
        extracted_item = {
            'title': item.get('title'),
            'description': item.get('description'),
            'keywords': item.get('keywords'),
            'category': item.get('category'),
            'tags': item.get('tags')
        }
        extracted_data.append(extracted_item)

    # Write the extracted data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(extracted_data, outfile, indent=4)


# Example usage
extract_data('article300Data.json', 'extracted_data.json')

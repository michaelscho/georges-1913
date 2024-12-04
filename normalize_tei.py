import lxml.etree as LET
import os
import re
from normalize_word import load_model, normalize_string

# Load the model, vocabulary, and device
model, vocab, device = load_model(repo_id="mschonhardt/georges-1913-normalization-model")

file = "F13.xml"
path = os.path.join(os.getcwd(), 'input', file)

# Open XML file in ./input/ using etree
tree = LET.parse(path)
root = tree.getroot()

# Get body
body = root.find(".//{http://www.tei-c.org/ns/1.0}body")

# Define tags to exclude from processing
EXCLUDED_TAGS = ['{http://www.tei-c.org/ns/1.0}label']  # Add more tags as needed

def clean_item(word):
    word = word.strip()
    word = word.replace("ęccl", "eccl")
    return word

def process_word(word):
    word = clean_item(word)
    #print(word)
    lower_word = word.lower()
    processed_word = normalize_string(lower_word, model, vocab, device)
    if word[0].isupper():
        processed_word = processed_word.title()
    #print(processed_word)
    return processed_word

def collect_tokens(element, tokens, position):
    # Check if the element is excluded
    tag = element.tag
    if tag in EXCLUDED_TAGS:
        return position  # Skip this element and its children

    # Tokenize the text of the element
    if element.text:
        position = tokenize_text(element.text, element, 'text', tokens, position)

    # Recursively collect tokens from child elements
    for child in element:
        position = collect_tokens(child, tokens, position)
        # Tokenize the tail text of the child
        if child.tail:
            position = tokenize_text(child.tail, child, 'tail', tokens, position)

    return position

def tokenize_text(text, node, text_type, tokens, position):
    for match in re.finditer(r'\b\w+\b|\W+', text):
        token_text = match.group()
        tokens.append({
            'text': token_text,
            'node': node,
            'text_type': text_type,
            'start': position,
            'end': position + len(token_text),
        })
        position += len(token_text)
    return position

def merge_split_words(tokens):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if re.match(r'\w+', token['text']):
            # Start a new merged token
            sub_tokens = [token]
            text = token['text']
            j = i + 1
            while j < len(tokens):
                next_token = tokens[j]
                # Check if next token is a word token and adjacent in text
                if re.match(r'\w+', next_token['text']) and next_token['start'] == token['end']:
                    # Merge next token
                    text += next_token['text']
                    sub_tokens.append(next_token)
                    token['end'] = next_token['end']
                    tokens[j]['merged'] = True
                    j += 1
                    token = next_token
                else:
                    break
            token = tokens[i]  # Reset to the initial token
            token['text'] = text
            token['sub_tokens'] = sub_tokens
            merged_tokens.append(token)
            i = j
        else:
            merged_tokens.append(token)
            i += 1
    # Filter out tokens that were merged
    merged_tokens = [t for t in merged_tokens if not t.get('merged', False)]
    return merged_tokens

def process_tokens(tokens):
    for token in tokens:
        if re.match(r'\w+', token['text']):
            token['processed_text'] = process_word(token['text'])
        else:
            token['processed_text'] = token['text']  # Non-word tokens remain unchanged

def map_tokens_to_nodes(tokens):
    node_texts = {}
    for token in tokens:
        processed_text = token['processed_text']
        if 'sub_tokens' in token:
            # Distribute processed_text to sub_tokens
            sub_tokens = token['sub_tokens']
            total_length = sum(len(t['text']) for t in sub_tokens)
            # Calculate cumulative positions
            positions = [0]
            for sub_token in sub_tokens:
                positions.append(positions[-1] + len(sub_token['text']))
            # Map positions to processed_text
            total_processed_length = len(processed_text)
            # Calculate positions in the processed text
            processed_positions = [int(round(p * total_processed_length / total_length)) for p in positions]
            # Adjust last position to ensure total length matches
            processed_positions[-1] = total_processed_length
            # Distribute processed text to sub-tokens
            for idx, sub_token in enumerate(sub_tokens):
                start = processed_positions[idx]
                end = processed_positions[idx + 1]
                processed_sub_text = processed_text[start:end]
                node = sub_token['node']
                text_type = sub_token['text_type']
                key = (id(node), text_type)
                if key not in node_texts:
                    node_texts[key] = ''
                node_texts[key] += processed_sub_text
        else:
            # Token is not merged, assign processed_text directly
            node = token['node']
            text_type = token['text_type']
            key = (id(node), text_type)
            if key not in node_texts:
                node_texts[key] = ''
            node_texts[key] += processed_text
    # Assign the reassembled texts back to the nodes
    for (node_id, text_type), text in node_texts.items():
        # Find the node by its id
        node = next(t['node'] for t in tokens if id(t['node']) == node_id)
        if text_type == 'text':
            node.text = text
        elif text_type == 'tail':
            node.tail = text

def process_element(element):
    # Start with position 0
    tokens = []
    collect_tokens(element, tokens, position=0)

    if tokens:
        tokens = merge_split_words(tokens)
        process_tokens(tokens)
        map_tokens_to_nodes(tokens)

process_element(body)

# Optionally, write the modified tree back to a file
output_path = os.path.join(os.getcwd(), 'output', file)
tree.write(output_path, pretty_print=True, encoding='UTF-8', xml_declaration=True)



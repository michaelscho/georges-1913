import lxml.etree as LET
import os
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

def clean_item(item):
    item = item.strip()
    item = item.replace("ęccl", "eccl")
    return item

def normalize_text(text):
    """
    Normalize a string of text, handling spaces correctly.
    """
    if not text:
        return text
    words = text.split()
    normalized_words = []
    for word in words:
        word = clean_item(word)
        if not word.isupper():
            normalized_text = normalize_string(word, model, vocab, device)
        else:
            word = word.lower()
            normalized_text = normalize_string(word, model, vocab, device)
            normalized_text = normalized_text.capitalize()
        normalized_words.append(normalized_text)
    return " ".join(normalized_words)

def normalize_node_text(node):
    """
    Recursively normalize the text and tail content of an XML node,
    ensuring proper inline spacing and handling of mixed content.
    """
    # Normalize main text
    if node.text:
        node.text = normalize_text(node.text)

    # Recursively normalize child nodes
    for child in node:
        normalize_node_text(child)

    # Normalize tail content without introducing extra spaces
    if node.tail:
        node.tail = normalize_text(node.tail)

# Apply normalization to all children of <body>
for child in body.iter():
    normalize_node_text(child)

# Write the modified XML to the output file
output_path = os.path.join(os.getcwd(), 'output', file)
tree.write(output_path, encoding='utf-8', xml_declaration=True)

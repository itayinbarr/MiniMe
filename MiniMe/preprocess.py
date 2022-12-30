import re
import string


# Before using the dataset, I need to preprocess the text.
def text_preprocess(path):
    # Open the text file in read mode
    file_path = './data/' + path
    with open(file_path, 'r') as f:
        # Read all lines of the file into a list
        lines = f.readlines()

    # Create a new list to store the modified lines
    new_lines = []

    # Iterate over each line in the list
    for line in lines:
        # Removing messages sent by other person
        if "Itay Inbar: " not in line:
            continue

        # Remove timestamp and name from my messages
        if ': ' in line:
            end_index = line.index(': ') + 1
        line = line[end_index + 1:]

        # Remove all kinds of audio recordings and non relevant media
        if 'omitted' in line:
            continue

        # Remove english words and numbers
        pattern = r'[0-9a-zA-Z]'
        line = re.sub(pattern, '', line)

        # Remove laughing sign in Hebrew
        if 'חחחחח' in line:
            continue

        line = line.translate(str.maketrans('', '', string.punctuation))
        # Add the modified line to the new list
        new_lines.append(line)

    # Save the preprocessed file
    with open("./data/" + path.split('.')[0] + " - processed.txt", 'w') as f:
        # Write the modified lines to the new file
        for line in new_lines:
            f.write(line)



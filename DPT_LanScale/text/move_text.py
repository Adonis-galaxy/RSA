import os
import shutil
from collections import Counter
import inflect
p_engine = inflect.engine()
import random

def remove_empty_lines_from_txt(folder):
    # Walk through folder
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            if file.endswith('.txt'):
                # Full path to the txt file in folder
                file_path = os.path.join(dirpath, file)

                # Read the file, remove empty lines, and overwrite the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Remove empty lines and strip whitespace characters
                # new_lines = [line for line in lines if line.strip() != ""]
                new_lines = [line.rstrip() + '\n' if line.rstrip() else '\n' for line in lines]

                # Write the non-empty lines back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

                print(f"Processed file: {file_path}")

def combine_repetitive_words(items):
# input, text_list: ['laptop', 'wall', 'tv', 'table', 'paper', 'tv', 'cardboard']
# output: string: "An image with a laptop, a wall, two tvs, a table, a paper, and a cardboard."
    # Use Counter to count occurrences of each item
    if len(items)==0:
        return "An image."
    if len(items)==1:
        print("An image with a " + items[0] +".", flush=True)
        return "An image with a " + items[0] +"."

    item_counts = Counter(items)

    # Initialize the sentence
    sentence = "An image with "

    # Convert the counts into a readable string format
    descriptions = []
    for item, count in item_counts.items():
        if count == 1:
            descriptions.append(f"a {item}")
        else:
            plural_word = p_engine.plural(item)
            descriptions.append(f"{p_engine.number_to_words(count)} {plural_word}")

    if len(descriptions)==0:
        return "An image."
    if len(descriptions)==1:
        print("An image with " + descriptions[0] +".", flush=True)
        return "An image with " + descriptions[0] +"."
    # Join the descriptions with commas and 'and' before the last item
    sentence += ", ".join(descriptions[:-1]) + ", and " + descriptions[-1] + "."
    return sentence

def append_content_to_matching_files(folderA, folderB):
    # Walk through folderA
    for dirpath, _, filenames in os.walk(folderA):
        for file in filenames:
            if file.endswith('.txt'):
                # Full path to the txt file in folderA
                file_path_A = os.path.join(dirpath, file)

                # Find corresponding file in folderB
                relative_path = os.path.relpath(file_path_A, folderA)  # Get relative path from folderA root
                file_path_B = os.path.join(folderB, relative_path)    # Map it to folderB

                # Check if the corresponding file exists in folderB
                if os.path.exists(file_path_B):
                    # Read content from file in folderA
                    with open(file_path_A, 'r', encoding='utf-8') as f_A:
                        content_A = f_A.read()
                        object_list=[]
                        for line in content_A.split("\n"):
                            # print(line)
                            line = line.split(" ")[0]
                            if line == "":
                                continue
                            object_list.append(line)
                        # print(object_list,flush=True)
                    sentence_list = []
                    for i in range(5):
                        random.shuffle(object_list)
                        # if len(object_list) > 1:
                        #     # Randomly select a subset of the list with at least one item
                        #     remaining_items = random.sample(object_list, k=random.randint(1, len(object_list)))
                        # else:
                        #     # If the list has only one item, keep it
                        #     remaining_items = object_list
                        # sentence = combine_repetitive_words(remaining_items)



                        # print(sentence, flush=True)
                        sentence = combine_repetitive_words(object_list)
                        sentence_list.append(sentence)
                        # print(sentence,flush=True)
                    # raise()


                    # Append content to file in folderB
                    with open(file_path_B, 'a', encoding='utf-8') as f_B:
                        for i in range(len(sentence_list)):
                            f_B.write(sentence_list[i] + "\n")

                    print(f"Appended content from {file_path_A} to {file_path_B}")
                else:
                    print(f"File {file} not found in {folderB}")

# Example usage
# appends the contents of the file from folder_source to the one in folder_dest
folder_source = 'seg_txt_panoptic'  # Replace with actual path to folderA
folder_dest = 'text_all'  # Replace with actual path to folderB

append_content_to_matching_files(folder_source, folder_dest)
# remove_empty_lines_from_txt(folderB)


# in the text file, mistral is line 1-5, vicuna is line 6-10, strctual text is 11-20
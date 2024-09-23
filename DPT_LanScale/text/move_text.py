import os
import shutil

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

                    # Append content to file in folderB
                    with open(file_path_B, 'a', encoding='utf-8') as f_B:
                        f_B.write("\n" + content_A)

                    print(f"Appended content from {file_path_A} to {file_path_B}")
                else:
                    print(f"File {file} not found in {folderB}")

# Example usage
# appends the contents of the file from folderA to the one in folderB
folderA = 'text_llava-v1.6-vicuna-7b_5captions'  # Replace with actual path to folderA
folderB = 'text_llava-v1.6-mistral-7b_5captions'  # Replace with actual path to folderB

# append_content_to_matching_files(folderA, folderB)
remove_empty_lines_from_txt(folderB)


# in the text file, mistral is line 1-5, vicuna is line 6-10
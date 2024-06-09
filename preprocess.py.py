import re


def clean_chat(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_lines = file.readlines()

    cleaned_lines = []
    for line in chat_lines:
        # Menyimpan tanggal, waktu, dan nama pengirim tanpa perubahan
        cleaned_lines.append(line.strip())

    with open(output_path, 'w', encoding='utf-8') as file:
        for line in cleaned_lines:
            file.write(f"{line}\n")


# Path ke file chat WhatsApp
input_file_path = 'chat.txt'
output_file_path = 'cleaned_chat.txt'

clean_chat(input_file_path, output_file_path)

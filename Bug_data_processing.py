from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"

def Start(data):
    inp = data

    # combine word, eg. AbcDef --> Abc & Def
    def combined_word(word):
        if word.isupper() == True:
            return word
        else:
            # regex [A-Z][a-z]* means any string starting
            # with capital character followed by many
            # lowercase letters
            words = re.findall('[A-Z][a-z]*', word)

            # Change first letter of each word into lower case
            result = ''
            for word in words:
                word = chr(ord(word[0]) + 32) + word[1:]
                result += word
                result += ' '
            return result

    # Tokenize sentences for description
    def tokenize_sentences(corpus):
        line = str(corpus)
        n = len(line)
        i = 0
        line_token = []
        lline = ''
        while (i < n):
            # if (line[i] !=" \" "):
            lline += line[i]

            if (i < n - 2):
                if (line[i] == '.') and (line[i + 2] == line[i + 2].upper()) and (line[i + 1].isdigit() == False):
                    line_token.append(lline)
                    lline = ''
            i += 1
        if (lline != ''):
            line_token.append(lline)
        return line_token

    # Data pre-processing
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # process the summary text
    text = inp['summary'].values
    text1_save = []
    for y1 in text:
        line_after = ''
        y1 = re.sub(r'\W', ' ', y1)  # remove puntuaction
        y1 = re.sub(r'\d', ' ', y1)  # remove digits
        words = y1.split()
        line = ''
        for word in words:
            word1 = word.lower()
            if word1 not in stop_words:
                if word.islower() == True:
                    word = word.capitalize()
                line += combined_word(word)
                line += ' '
        words = line.split()
        for w1 in words:
            if (len(w1) != 1):  # if w1 is a character then delete
                w1 = ps.stem(w1)
                line_after += ' '
                line_after += w1
        text1_save.append(line_after.lower())

    # precess the description text
    text = inp['description'].values
    text2_save = []
    for row in text:
        if (type(row)==str):
            x = tokenize_sentences(row)
            line_after = ''
            for y1 in x:
                y1 = re.sub(r'\W', ' ', y1)  # remove puntuaction
                y1 = re.sub(r'\d', ' ', y1)  # remove digits
                line = ''
                words = y1.split()
                line = ''
                for word in words:
                    word1 = word.lower()
                    if word1 not in stop_words:
                        if word.islower() == True:
                            word = word.capitalize()
                        line += combined_word(word)
                        # line+=word
                        line += ' '

                words = line.split()

                for w1 in words:
                    if (len(w1) != 1):  # if w1 is a character then delete
                        w1 = ps.stem(w1)
                        line_after += ' '
                        line_after += w1
                line_after += '.'
            text2_save.append(line_after.lower())
        else:
            text2_save.append('')

    x = pd.DataFrame({'summary': text1_save, 'description': text2_save})
    # inp['description'] = text_save
    x.to_csv(BUG_CSV)
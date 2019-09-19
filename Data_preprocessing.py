from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pandas as pd
import re

def Start():
    inp = pd.read_csv('Training_Test\AspectJ_process.csv')

    def combined_word(word):

        if word.isupper() == True:
            return word
        else:
            # regex [A-Z][a-z]* means any string starting
            # with capital character followed by many
            # lowercase letters
            words = re.findall('[A-Z][a-z]*', word)

            # Change first letter of each word into lower
            # case
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
    '''
    # Summary processing
    x = inp['summary'].values
    x_save = []
    for x1 in x:
        x1 = re.sub(r'\W', ' ', x1)  # remove puntuaction
        x1 = re.sub(r'\d', ' ', x1)  # remove digits
        line = ''
        words = x1.split()
        line = ''
        for word in words:
            word1 = word.lower()  # không bị mất từ viết hoa để sau đó còn combin từ
            if word1 not in stop_words:
                if word.islower() == True:
                    word = word.capitalize()
                line += combined_word(word)
                # line += word
                line += ' '

        words = line.split()
        line_after = ''
        for w1 in words:
            if (len(w1) != 1):  # if w1 is a character then delete
                w1 = ps.stem(w1)
                line_after += w1
                line_after += ' '
        x_save.append(line_after.lower())
    inp['summary'] = x_save'''

    y = inp['description'].values
    y_save = []
    for row in y:
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
            y_save.append(line_after.lower())
        else:
            y_save.append('')


    inp['description'] = y_save
    inp.to_csv('Training_Test\AspectJ_process.csv')
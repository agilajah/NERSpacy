#!/usr/bin/env python
# coding: utf8

#!/usr/bin/env python
# coding: utf8
# example: python NERSpacy_test.py -i /media/agilajah/DATA/Repos/prosa/NERSpacy/models/12feb2018141420 -id /media/agilajah/DATA/Repos/prosa/NERSpacy/data/out

from __future__ import unicode_literals, print_function

import plac
import random
import spacy
from pathlib import Path

ref_data = []
word_data = []

@plac.annotations(
    input_dir=("Input model directory", "option", 'i', Path), 
    input_data=("Input ref and test data directory", "option", "id", Path))
def main(input_dir=None, input_data=None):
        # loading ref data and test data
        word = str(input_data) + '/word.txt'
        ref = str(input_data) + '/ref.txt'
        pred = str(input_data) + '/pred.txt'
        with open(word, "rt") as in_file:
            data = in_file.readlines()
            for line in data:
                word_data.append(line)
        
        with open(ref, "rt") as in_file:
            data = in_file.readlines()
            for line in data:
                ref_data.append(line)

        # test the saved model
        print("Loading from", input_dir)
        nlp = spacy.load(input_dir)
        counter = 0
        for text in word_data:
            temp = 'PRED: '
            doc = nlp(text[7:])
            # print("WORD: " + text[7:])
            # print("REF: " + ref_data[counter])
            for ent in doc.ents:
                temp = temp + ent.label_ + "    "
                # print(ent.label_, ent.text)
            # print("PRED: " + temp)
            with open(pred, "a", encoding="utf8") as myfile:
                myfile.write(text)
                myfile.write(ref_data[counter])
                myfile.write(temp)
                myfile.write('\n\n')
                
            counter = counter + 1

if __name__ == '__main__':
    plac.call(main)
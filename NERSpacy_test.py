#!/usr/bin/env python
# coding: utf8

#!/usr/bin/env python
# coding: utf8
# example: python NERSpacy_test.py -i D:\Repos\prosa\NERSpacy 

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy

@plac.annotations(
    input_dir=("Input directory", "option", 'i', Path))
def main(input_dir=None):
        # test the saved model
        test_text = ''
        print("Loading from", input_dir)
        nlp = spacy.load(input_dir)
        doc = nlp(test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

if __name__ == '__main__':
    plac.call(main)
#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy

@plac.annotations(
    output_dir=("Optional output directory", "option", "o", Path))
def main(output_dir=None):
        # test the saved model
        test_text = 'Presiden terpilih  JOkO WidOdO   mengungkapkan pihaknya tidak akan membedakan spesifikasi kandidat menteri yang diusung Oleh partai pOlitik pengusung maupun pendukung.'
        print("Loading from", output_dir)
        nlp = spacy.load(output_dir)
        doc = nlp(test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

if __name__ == '__main__':
    plac.call(main)
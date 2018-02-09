#!/usr/bin/env python
# coding: utf8
# example: python NERSpacy_train.py -nm test -o D:\Repos\prosa\NERSpacy 
"""Example of training an additional entity type
This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.
The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
from helper.data import *
import spacy


# new entity label

LABEL_BEGINNING_PLC = 'B-PLC'
LABEL_INSIDE_PLC = 'I-PLC'
LABEL_BEGINNING_PPL = 'B-PPL'
LABEL_INSIDE_PPL = 'I-PPL'
LABEL_BEGINNING_EVT = 'B-EVT'
LABEL_INSIDE_EVT = 'I-EVT'
LABEL_BEGINNING_IND = 'B-IND'
LABEL_INSIDE_IND = 'I-IND'
LABEL_BEGINNING_FNB = 'B-FNB'
LABEL_INSIDE_FNB = 'I-FNB'
LABEL_OTHER = 'O'


# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
# TRAIN_DATA = [
    # ("Presiden terpilih  JOkO WidOdO   mengungkapkan pihaknya tidak akan membedakan spesifikasi kandidat menteri yang diusung Oleh partai pOlitik pengusung maupun pendukung.", {
    #     'entities': [(0, 8, 'O'), (9, 17, 'O'), (19, 23, 'B-PPL'), (24, 30, 'I-PPL'), (33, 46, 'O'), (53, 61, 'O'), (62, 67, 'O'), (68, 72, 'O'), (73, 83, 'O')]
    # }),
    #     ("Presiden terpilih  JOkO WidOdO   mengungkapkan pihaknya tidak akan membedakan spesifikasi kandidat menteri yang diusung Oleh partai pOlitik pengusung maupun pendukung.", {
    #     'entities': [(78, 89, 'O'), (90, 98, 'O'), (99, 106, 'O'), (107, 111, 'O'), (112, 119, 'O'), (120, 124, 'O'), (125, 139, 'O'), (140, 149, 'O'), (150, 156, 'O'), (157, 166, 'O'), (166, 167, 'O')]
    # }),
# ]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    # add new entity label to entity recognizer
    ner.add_label(LABEL_BEGINNING_PLC)
    ner.add_label(LABEL_INSIDE_PLC)
    ner.add_label(LABEL_BEGINNING_PPL)
    ner.add_label(LABEL_INSIDE_PPL)
    ner.add_label(LABEL_BEGINNING_EVT)
    ner.add_label(LABEL_INSIDE_EVT)
    ner.add_label(LABEL_BEGINNING_IND)
    ner.add_label(LABEL_INSIDE_IND)
    ner.add_label(LABEL_BEGINNING_FNB)
    ner.add_label(LABEL_INSIDE_FNB)
    ner.add_label(LABEL_OTHER)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'Presiden terpilih  JOkO WidOdO   mengungkapkan pihaknya tidak akan membedakan spesifikasi kandidat menteri yang diusung Oleh partai pOlitik pengusung maupun pendukung.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
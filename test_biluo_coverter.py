from spacy.gold import biluo_tags_from_offsets
import spacy

nlp = spacy.load('en')
doc = nlp(u'I like London.')
entities = [(7, 13, 'LOC')]
tags = biluo_tags_from_offsets(doc, entities)
print(tags)
assert tags == ['O', 'O', 'U-LOC', 'O']
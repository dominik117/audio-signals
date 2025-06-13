from nltk.corpus import wordnet as wn

import nltk
#nltk.download('wordnet')


synonyms = [] 
antonyms = [] 

word = "car"

syns = wn.synsets(word) 

# An example of a synset: 
print(syns[0].name()) 

# Just the word: 
print(syns[0].lemmas()[0].name()) 

# Definition of that first synset: 
print(syns[0].definition()) 

# Examples of the word in use in sentences: 
print(syns[0].examples()) 

# Synonmyes and antonyms
for syn in wn.synsets(word): 
	for l in syn.lemmas(): 
		synonyms.append(l.name()) 
		if l.antonyms(): 
			antonyms.append(l.antonyms()[0].name()) 

print(set(synonyms)) 

print(set(antonyms)) 

import csv
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
    
'''
Read a textfile and return a list of the words in the file. The file should be structured so that 
each word is on a new line.
'''
def readWordList(textFile):
    words = []
    with open(textFile, 'r') as f:
        for line in f:
            term = line.replace('\n', '')
            words.append(term)
    return words


'''
Helper function so that other functions can take either a dictionary or a list as input.
'''
def getIterator(input):
    if type(input) == dict:
        return input.values()
    elif type(input) == list:
        return input
    else:
        raise TypeError('Input must be a dictionary or a list')

def checkIdx(idx, lcsh): 
    try: 
        lcsh[idx]
        return True
    except KeyError: 
        return False
    
'''
Get the year a LCSH was added to the online database
'''
def yearAdded(term):
    return int(term['yearNew'][:4])

'''
Get the distribution of LCSHs by year added to the online database. 
'''
def getTimeline(lcsh):
    years = [y for y in range(1986, 2024)]
    timeline = {y:0 for y in years}
    # check if the input is a dictionary or a list
    lcshIterator = getIterator(lcsh)
    for term in lcshIterator:
        yearNew = yearAdded(term)
        if yearNew != 2024:
            timeline[yearNew] += 1
        if term['yearDep'] is not None:
            yearDep = int(term['yearDep'][:4])
            if yearDep != 2024:
                timeline[yearDep] -= 1
    return timeline

'''
Collects the number of terms in the LCSH that are deprecated and returns a dictionary of these terms.
'''
def getDeprecated(lcsh):
    count = 0
    dep = {}
    for idx, heading in lcsh.items():
        if heading['yearDep'] is not None:
            dep[idx] = heading
            count += 1
    return count, dep

'''
Remove deprecated LCSHs that were created and removed within the span of a year. 
'''
def pruneDeprecated(lcsh):
    pruned = {}
    for idx, term in lcsh.items():
        if term['yearDep'] is None:
            pruned[idx] = term
        else:
            if int(term['yearDep'][:4]) -int(term['yearNew'][:4]) >= 1:
                pruned[idx] = term
    return pruned

'''
Count the types of LCSHs in a dataset of headings
'''
def countTypes(lcsh):
    kinds = {}
    lcshIterator = getIterator(lcsh)
    for heading in lcshIterator:
        kind = heading['type']
        if kind in kinds:
            kinds[kind] += 1
        else:
            kinds[kind] = 1
    return kinds

'''
Collect all LCSH that are simple terms. This means that they are not subdivisions or complex headings.
'''
def getSimpleTerms(lcsh):
    simple = {}
    simpleTypes = ['Topic', 'Geographic', 'CorporateName', 'FamilyName', 'Title', 
                   'ConferenceName', 'PersonalName']
    for idx, term in lcsh.items():
        if term['type'] in simpleTypes:
            simple[idx] = term
    return simple

'''
Get the the top-level LCC category breakdown (the first letter) of a set of LCSHs. Not all LCSHs have LCC categories.
raw is a boolean that determines whether the function returns the raw number of terms in 
each category or the proportion of terms in each category.
The function returns a dictionary of the categories and the total number of LCSHs that have LCC categories.
'''
def getLCCBreakdown(lcsh, raw=False):
    lccTerms = {idx: term for idx, term in lcsh.items() if term['lcc'] is not None}
    total = len(lccTerms)
    cats = {}
    lcshIterator = getIterator(lccTerms)
    for term in lcshIterator:
        cat = term['lcc'][0]
        if raw:
            val = 1
        else:
            val = 1/total
        if cat in cats:
            cats[cat] += val
        else:
            cats[cat] = val
    return cats, total

'''
Code to clean up the LCSH headings. This function removes any unwanted characters and makes all the words lowercase.
'''
def clean(word):
    return word.lower().replace('(', '').replace(')', '').replace("'", " '").replace(',', '').replace('_', '').replace('-', ' ')


def head2Ngram(head, n):
    if 'Cooking (' in head and head[:7] == 'Cooking':
        head = head.replace('(', '').replace(')', '')
    # if the term is a law term then I want to keep the brackets as they are important for the meaning of the term.
    # For example Women judges (Islamic law) becomes Women judges in Islamic law.
    elif 'law)' in head:
        head = re.sub(r'\((.*?)\)', r'in \1', head)
    # If the brackets occur at the end of the term and it does not fall under any special cases then they are removed. 
    elif '(' in head and head.index(')') == len(head)-1:
        head = re.sub(r'\(.*?\)', r'', head)
    # For now I'm not including terms where thr brackets occur in the middle of the term as they are tricky to deal with.
    elif '(' in head and head.index(')') != len(head)-1:
        return ''
    head = head.replace('-', ' ').replace("'", " '")
    # Check if the term contains the gendered term of interest
    if len(head.split(' ')) <= n:
        # Format the term so that the last word comes first if they are separated by a comma. 
        # This is so that the n-grams are more easily comparable.
        if head.count(',') == 1:
            parts = head.split(',')
            part1 = parts[0].strip()
            part2 = parts[1].strip()
            head = part2 + ' ' + part1
            return head
        elif head.count(',') > 1:
            head = re.sub(r"(\w+(( | ')\w+)*), ((\w+)( \w+)*)", r'\4 \1', head).replace(',', '')
            return head
        else:
            return head
    else:
        return ''
    

'''
Convert a list of LCSHs to a list of n-grams. The function returns a list of n-grams, a list of the years 
the n-grams were added, and a list of the indices of the n-grams.

Only LCSHs with less than n words are included in the n-grams.

In most cases the function will remove any words in parentheses however there are some exceptions.

If genTerm is True, then only LCSHs that contain the gendered term are included in the n-grams. This is because 
it is possible the gendered terms occurs inside the brackets and is thus removed. 
'''
def getNgrams(terms, n, genTerm=False, synonym=False):
    nGrams = []
    indices = []
    yrs = []
    for idx, term in terms.items():
        nGram = head2Ngram(term['heading'], n).lower().strip()
        if nGram == '' or (genTerm and term['genTerm'] not in nGram):
            continue
        else:
            if synonym and term['synonyms'] is not None:
                for syn in term['synonyms']:
                    variant = head2Ngram(syn, n).lower().strip()
                    if variant != '' and variant not in nGram:
                        nGram += "," + variant
            nGrams.append(nGram)
            yrs.append(yearAdded(term))
            indices.append(idx)
    return nGrams, yrs, indices

'''
Read frequency data from a csv file. The csv file should have the following format:
- The first row should be the column headings. 
- The first column should be the index of the term.
- The second column should be the heading of the term.
- The third column should be the year the term was added to the LCSH.
- The remaining columns should be the frequency data.

The function returns a numpy array of the frequency data, a list of the years, a list of the headings, and a list of the indices.
'''
def getNgramData(csvName, goodIndices=None):
    freqs, heads, yrs, indices = [], [], [], []
    with open(csvName, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if goodIndices is not None and row[0] not in goodIndices:
                continue
            indices.append(row[0])
            heads.append(row[1])
            yrs.append(int(row[2]))
            freqs.append(np.array(row[3:], dtype=float))
    return np.row_stack(freqs), yrs, heads, indices


'''
Find pairs of terms that are gendered. The function takes in dictionaries and returns a list of pairs
that only differ by the use of a word referring to women or a word referring to men. All the other words
in the pair must be the same.
'''
def findPairs(mTerms, wTerms, pairs):
    pairedTerms = []
    indicesW, indicesM = [], []
    mIterator, wIterator = getIterator(mTerms), getIterator(wTerms)
    mHeads = [t['heading'] for t in mIterator]
    wHeads = [t['heading'] for t in wIterator]
    for i in range(len(pairs)):
        wordM, wordW = pairs[i]
        replaced = {}
        for i, head in enumerate(wHeads):
            wHead = head.lower()
            if wordW in wHead:
                replaced[wHead.replace(wordW, '')] = i
        for i, head in enumerate(mHeads):
            mHead = head.lower()
            if wordM in mHead:
                genTerm = mHead.replace(wordM, '')
                if genTerm in replaced.keys() and i not in indicesM and replaced[genTerm] not in indicesW:
                    pairedTerms.append((list(mTerms.items())[i],list(wTerms.items())[replaced[genTerm]]))
                    indicesM.append(i)
                    indicesW.append(replaced[genTerm])
    return pairedTerms

'''
Collect all terms in the LCSH that contain a words from a list. The function returns a dictionary of the terms.
The parameter tag is the key that the word is stored under in the dictionary representing the LCSH. 
Returns a dictionary of the terms that contain the words in the list.
'''
def getTermsWithWords(lcsh, wordList, tag):
    headings = {}
    for idx, term in lcsh.items():
        flag = False
        # a lot of specific cases of terms that I don't want to include in the analysis. 
        if (term['type'] == 'Topic' and term['lang'] == 'en' 
            and '(Fictitious character' not in term['heading']
            and '(Symbolic character' not in term['heading']
            and '(Legendary character' not in term['heading']
            and '(Game)' not in term['heading']
            and '(International relations)' not in term['heading']
            and 'word)' not in term['heading']
            and '(Legend)' not in term['heading']
            and '(Miracle)' not in term['heading']
            and '(Race horse)' not in term['heading']
            and '(Horse)' not in term['heading']
            and '(Dog)' not in term['heading']
            and ' mythology)' not in term['heading']
            and 'deities)' not in term['heading']
            and '(Parable)' not in term['heading']
            and '(Tale)' not in term['heading']
            and '(Nickname)' not in term['heading']
            and '(Statue)' not in term['heading']
            and 'deity)' not in term['heading']
            and '(Art)' not in term['heading']
            and 'locomotives)' not in term['heading']
            and '(Imaginary' not in term['heading']):
            head = clean(term['heading'])
            words = [w for w in head.split(' ')]
            for w in wordList:
                if w in words:
                    term[tag] = w
                    term['baseForm'] = head
                    flag = True
                    break
        # it seems that in the LCSH that 'gay people' is the more general term with 'gay men' and 'lesbians' being the specific
        # gendered subsets. The following if case makes sure that my method doesn't pick up terms that refer to both lesbians and gay people.
        # if 'lesbian' in words and 'gay' in words:
        #     continue
        if flag:
            headings[idx] = term
    return headings

'''
Get all terms that mention both men and women.
'''
def getAmbiguous(mTerms, wTerms):
    mHeads = [idx for idx in mTerms.keys()]
    wHeads = [idx for idx in wTerms.keys()]
    return list(set(mHeads).intersection(wHeads))

'''
Get terms that contain a specific phrase. 
For example, 'lesbian and gay' is a phrase that is picked up by my method when looking for terms for women, 
however this phrase is not specific to women. 
'''
def getTermsWithPhrase(lcsh, phrases):
    indices = []
    for idx, term in lcsh.items():
        for phrase in phrases:
            if phrase in term['heading'].lower():
                indices.append(idx)
                break
    return indices


def pruneAmbiguous(mTerms, wTerms, ambiguous):
    with open('Data/LCSH/ambiguous-lcsh.txt', 'w') as f:
        for idx in ambiguous:
            if idx in mTerms and idx in wTerms:
                f.write(f'{mTerms[idx]['heading']}\n')
                del mTerms[idx]
                del wTerms [idx]
            elif idx in wTerms:
                f.write(f'{wTerms[idx]['heading']}\n')
                del wTerms [idx]
            elif idx in mTerms:
                f.write(f'{mTerms[idx]['heading']}\n')
                del mTerms[idx]

def checkParents(term):
    hasParents = False
    if term['bt'] is not None:
        parents = [idx for idx in term['bt'] if  idx.strip()]
        if parents != []:
            hasParents = True
    return hasParents 

def getParents(term, lcsh):
    if term['bt'] is not None:
        parents = [lcsh[idx] for idx in term['bt'] if idx.strip()]
    else:
        parents = []
    return parents

def getSiblings(term, parent, lcsh):
    heads = [lcsh[idx]['heading'] for idx in parent['nt'] if idx.strip() and checkIdx(idx, lcsh)]
    if term['heading'] not in heads:
        raise ValueError('Parent LCSH is not the parent of the selected LCSH')
    elif len(heads) == 1:
        return []
    else:
        return [lcsh[idx] for idx in parent['nt'] if idx.strip() and checkIdx(idx, lcsh) and lcsh[idx]['heading'] != term['heading']]    
    

def countGenderedParents(terms, lcsh):
    hasGendered = 0
    onlyGendered = 0
    for term in terms.values():
        parents = getParents(term, lcsh)
        genderedParents = getGenderedParent(term, parents)
        if len(genderedParents) > 0:
            hasGendered += 1
            if len(genderedParents) == len(parents):
                onlyGendered += 1
    return hasGendered, onlyGendered


def getGenderedParent(term, parents):
        genParents = []
        genWord = term['genTerm'] 
        for parent in parents:
            if genWord in parent['heading'].lower():
                genParents.append(parent)
        return genParents

def getGenericParent(term, parents):
    genericParents = []
    ignoreWords = ["and", "in", "for", "'s"]
    for parent in parents:
        genWord = term['genTerm'] 
        if genWord in parent['heading'].lower():
            continue
        parentWords = [lemmatizer.lemmatize(w) for w in clean(parent['heading']).lower().split(' ') if w not in ignoreWords and w.strip() != '']
        childWords = [lemmatizer.lemmatize(w) for w in clean(term['heading']).lower().split(' ') if w != genWord and w not in ignoreWords and w.strip() != '']
        sortParent, sortChild = sorted(parentWords), sorted(childWords)
        if sortParent == sortChild:
            genericParents.append(parent)
        elif sortParent == sorted(sortChild + ['person']) or sortParent == sorted(sortChild + ['people']): 
            genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['child']):
        #     genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['spouse']):
        #     genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['parent']):
        #     genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['sibling']):
        #     genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['partner']):
        #     genericParents.append(parent)
        elif sortParent == sorted(sortChild + ['adult']):
            genericParents.append(parent)
        elif sortParent == sorted(sortChild + ['worker']):
            genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['grandchild']):
        #     genericParents.append(parent)
        # elif sortParent == sorted(sortChild + ['grandparent']):
        #     genericParents.append(parent)
        elif sortParent == sorted(sortChild + ['human', 'being']):
            genericParents.append(parent)
        elif sorted(sortParent + ['owned']) == sortChild:
            genericParents.append(parent)
    return genericParents


def cleanLCSH(lcsh):
    cleaned = {}
    ignoreWords = ["and", "in", "for", "'s"]
    for idx, term in lcsh.items():
        words = [lemmatizer.lemmatize(w) for w in clean(term['heading']).lower().split(' ') if w not in ignoreWords and w.strip() != '']
        cleaned[idx] = sorted(words)
    return cleaned

def getGeneric(term, cleanedLCSH):
    ignoreWords = ["and", "in", "for", "'s"]
    genWord = term['genTerm'] 
    childWords = [lemmatizer.lemmatize(w) for w in clean(term['heading']).lower().split(' ') if w != genWord and w not in ignoreWords and w.strip() != '']
    sortChild = sorted(childWords)
    if sortChild in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sortChild)]
    elif sorted(sortChild + ['person']) in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sorted(sortChild + ['person']))]
    elif sorted(sortChild + ['people']) in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sorted(sortChild + ['people']))]
    elif sorted(sortChild + ['adult']) in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sorted(sortChild + ['adult']))]
    elif sorted(sortChild + ['worker']) in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sorted(sortChild + ['worker']))]
    elif sorted(sortChild + ['human', 'being']) in cleanedLCSH.values():
        return list(cleanedLCSH.keys())[list(cleanedLCSH.values()).index(sorted(sortChild + ['human', 'being']))]
    else:
        return ''
# def getGeneric(term, parent)
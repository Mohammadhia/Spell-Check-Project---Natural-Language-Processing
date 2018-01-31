import math, collections

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 1)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.total -= self.unigramCounts[token] #Subtract original count of
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.total += self.unigramCounts[token] #Accounts for previous counts of the word in the unigram
                                                #(because smoothing is involved with a default value of 1,
                                                # adding 1 alone does not account for the default value)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of
    the sentence using your language model. Use whatever data you computed in
    train() here.  """

    ####COMMENTED OUT CODE BELOW IS NOT USED. JUST AN EXAMPLE OF INCORRECT IMPLEMENTATION OF LAPLACE SMOOTHING#################
    #for token in sentence:  #Add 1 smoothing to all words that were not originally in the training set
    #  count = self.unigramCounts[token]
    #  if(count == 0):
    #    self.unigramCounts[token] = self.unigramCounts[token] + 1
    #    self.total += 1
    #for token in sentence:  #Add 1 smoothing to the words ORIGINALLY IN the training set
    #  count = self.unigramCounts[token]
    #  if(count > 0):
    #    self.unigramCounts[token] = self.unigramCounts[token] + 1
    #    self.total += 1
    ####COMMENTED OUT CODE ABOVE IS NOT USED. JUST AN EXAMPLE OF INCORRECT IMPLEMENTATION OF LAPLACE SMOOTHING#################

    score = 0.0
    for token in sentence:
      count = self.unigramCounts[token]
      if count > 0:
        score += math.log(count)
        score -= math.log(self.total)
      else:
        score = float('-inf')  #Smoothing should have taken place outside this for loop, above
    return score

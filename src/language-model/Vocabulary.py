class Vocabulary:
    def __init__(self, vocabularyFilename=None):
        self.word2int = {}
        self.int2word = {}
        self.nextID = -1
        self.wordFrequencies = {}

        self.addSymbol("<s>")
        self.addSymbol("</s>")
        self.addSymbol("<unk>")

        if vocabularyFilename:
            # Initialize the vocabulary mappings
            vocabularyFile = open(vocabularyFilename, 'r')
            for line in vocabularyFile:
                self.addSymbol(line.strip())

            vocabularyFile.close()

    def unknownWordID(self):
        return self.word2int["<unk>"]

    def unknownWordString(self):
        return "<unk>"

    def size(self):
        return self.nextID + 1

    def addSymbol(self, word):
        if word not in self.word2int:
            self.nextID += 1
            self.word2int[word] = self.nextID
            self.int2word[self.nextID] = word

            self.wordFrequencies[self.nextID] = 1
        else:
            wordID = self.word2int[word]
            self.wordFrequencies[wordID] += 1

        return self.word2int[word]

    def symbol(self, ID):
        if ID not in self.int2word:
            return self.unknownWordString()
        else:
            return self.int2word[ID]

    def index(self, word):
        if word not in self.word2int:
            return self.unknownWordID()
        else:
            return self.word2int[word]

    def getWordFrequencyWordID(self, wordID):
        if wordID not in self.int2word:
            return 0
        else:
            return self.wordFrequencies[wordID]

    def getWordFrequencyWordString(self, word):
        wordID = self.index(word)
        return self.getWordFrequencyWordID(wordID)

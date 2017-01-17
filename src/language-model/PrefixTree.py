class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.count = 0
        self.numberOfFollowingContexts = 0

    def recursiveAddNGram(self, nGram):
        # nGram: vector of word IDs (integers)
        self.count += 1

        if len(nGram) == 0:
            return

        # len(nGram) != 0
        nextWordID = nGram[0]
        if nextWordID not in self.children:
            self.children[nextWordID] = PrefixTreeNode()
            self.numberOfFollowingContexts += 1

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        childNode.recursiveAddNGram(nGram[1:])

    def getNGramCount(self, nGram):
        # nGram: vector of word IDs (integers)

        if len(nGram) == 0:
            return self.count

        nextWordID = nGram[0]
        if nextWordID not in self.children:
            return 0

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        return childNode.getNGramCount(nGram[1:])

    def getNGramNode(self, nGram):
        if len(nGram) == 0:
            return self

        nextWordID = nGram[0]
        if nextWordID not in self.children:
            return None

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        return childNode.getNGramNode(nGram[1:])
class PrefixTreeNode:
    def __init__(self):
        self.children = None
        self.count = 0

    def recursiveAddNGram(self, nGram):
        # nGram: vector of word IDs (integers)
        self.count += 1

        if len(nGram) == 0:
            return

        if self.children == None:
            self.children = {}

        # len(nGram) != 0
        nextWordID = nGram[0]
        if nextWordID not in self.children:
            self.children[nextWordID] = PrefixTreeNode()

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        childNode.recursiveAddNGram(nGram[1:])

    def getNGramCount(self, nGram):
        # nGram: vector of word IDs (integers)

        if len(nGram) == 0:
            return self.count

        if self.children == None:
            self.children = {}

        nextWordID = nGram[0]
        if nextWordID not in self.children:
            return 0

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        return childNode.getNGramCount(nGram[1:])

    def getNGramNode(self, nGram):
        if len(nGram) == 0:
            return self

        if self.children == None:
            self.children = {}

        nextWordID = nGram[0]
        if nextWordID not in self.children:
            return None

        # Connection to the next node already exists
        childNode = self.children[nextWordID]
        return childNode.getNGramNode(nGram[1:])

    def subtreeSize(self):
        if self.children != None:
            return 1 + sum([child.subtreeSize() for id, child in self.children.items()])
        else:
            return 1

    def getChildren(self):
        if self.children != None:
            return self.children
        else:
            return {}

    def getNumberOfChildren(self):
        if self.children != None:
            return len(self.children)
        else:
            return 0


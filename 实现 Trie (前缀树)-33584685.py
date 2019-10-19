class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.node = {}        

    def insert(self, word: str) -> None:
        if len(word) == 0:
            return
        """
        Inserts a word into the trie.
        """
        _node = self.node
        _prev = None
        for c in word:
            if (c,0) in _node:
                _prev = _node
                _node = _node[(c,0)]
                
            elif (c,1) in _node:
                _prev = _node
                _node = _node[(c,1)]
            else:
                _prev = _node
                _node[(c,0)] = {}
                _node = _node[(c,0)]
        
        if (word[-1],0) in _prev:
            _prev[(word[-1],1)] = _prev[(word[-1],0)]
            del _prev[(word[-1],0)]
            

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        _node = self.node
        prev = None
        for c in word:
            if (c,0) in _node:
                prev = _node
                _node = _node[(c,0)]
            elif (c,1) in _node:
                prev = _node
                _node = _node[(c,1)]
            else:
                return False
        
        if (word[-1],1) in prev:
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        _node = self.node
        for c in prefix:
            if (c,0) in _node:
                _node = _node[(c,0)]
            elif (c,1) in _node:
                _node = _node[(c,1)]
            else:
                return False
        
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
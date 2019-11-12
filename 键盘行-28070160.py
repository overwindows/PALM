class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        # init
        mat = [['q','w','e','r','t','y','u','i','o','p'],['a','s','d','f','g','h','j','k','l'],['z','x','c','v','b','n','m']]
        alpha = {}
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                alpha[mat[i][j]] = i
        
        ret = []
        for word in words:
            s = list(word.lower())
            same = True
            for i in range(len(s)-1):
                if alpha[s[i]] != alpha[s[i+1]]:
                    same = False
                    break
            if same:
                ret.append(word)
        
        return ret
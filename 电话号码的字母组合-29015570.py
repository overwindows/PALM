class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        tel = [[],[],'abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        
        def combo(alphas, res):
            r = []
            for alpha in alphas:
                if not res:
                    r.append(alpha)
                for x in res:
                    r.append(x+alpha)
            return r
            
        
        res = []
        for d in digits:
            res = combo(tel[int(d)],res)
        
        res.sort()
        return res
        
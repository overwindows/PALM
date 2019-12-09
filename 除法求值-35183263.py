class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        d = {}
        if len(equations) == 0:
            return None
        ix = 0
        for e0,e1 in equations:
            if e0 not in d:
                d[e0] = ix
                ix += 1
            
            if e1 not in d:
                d[e1] = ix
                ix += 1
        
        n = len(d.keys())
        #print(n)
        mat = [[-1.0]*n for _ in range(n)]
        
        for i in range(n):
            mat[i][i] = 1.0
        
        for i in range(len(equations)):
            x,y = equations[i]
            #print(values[i])
            mat[d[x]][d[y]] = values[i]
            mat[d[y]][d[x]] = 1.0/values[i]
        
        loop = True
        
        #while loop:
            #loop = False
        for _ in range(n):
            for i in range(n):
                for j in range(n):
                    if mat[i][j] == -1.0:
                        for k in range(n):
                            if mat[i][k] != -1.0 and mat[k][j] != -1.0:
                                mat[i][j] =  mat[i][k] * mat[k][j]
                                mat[j][i] = 1.0/mat[i][j]
        
        ret = []
        for a,b in queries:
            if a in d and b in d:
                ret.append(mat[d[a]][d[b]])
            else:
                ret.append(-1.0)
        return ret
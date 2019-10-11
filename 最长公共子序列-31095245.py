class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1 = len(text1)
        len2 = len(text2)
        
        if len1 == 0 or len2 == 0:
            return 0
        
        #print(len1, len2)
        mat = [[0] * len2 for _ in range(len1)]
        
        for i in range(len1):
            for j in range(len2):
                if text1[i] == text2[j]:
                    mat[i][j] = 1
        
        _max = 0
        #print(mat)
        for i in range(len1):
            for j in range(len2):
                if i>0 and j>0:
                    if mat[i][j]:
                        mat[i][j] += mat[i-1][j-1]
                    else:
                        mat[i][j] = max(mat[i-1][j], mat[i][j-1])
                elif j > 0:
                    mat[i][j] = max(mat[i][j-1],mat[i][j])
                elif i > 0:
                    mat[i][j] = max(mat[i-1][j],mat[i][j])

                #print(mat)
                _max = max(mat[i][j],_max)
        
        
        return _max
                    
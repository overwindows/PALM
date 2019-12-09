import pprint

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)
        
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
        
        if len1 == 1 and len2 == 1:
            if word1 == word2:
                return 0
            else:
                return 1
        
        mat = [[sys.maxsize] * (len2+1) for _ in range(len1+1)]
        
        #for i in range(len1):
        #    for j in range(len2):
        #        mat[i][j] = 1-int(word1[i]==word2[j])
        
        mat[0][0] = 0
        
        for i in range(len1+1):
            for j in range(len2+1):
                #if i==0 and j==0:
                #    if word1[0] == word2[0]:
                #        mat[0][0] = 0
                #    else:
                #        mat[0][0] = 1
                if i == len1 or j == len2:
                    if i == len1 and j < len2:
                        mat[len1][j+1] = min(mat[len1][j]+1, mat[len1][j+1])
                    
                    if j == len2 and  i < len1:
                        mat[i+1][len2] = min(mat[i][len2]+1, mat[i+1][len2])
                    continue
                    
                if word1[i] != word2[j]:
                    #replace
                    mat[i+1][j+1] = min(mat[i][j]+1, mat[i+1][j+1])       
                    #insert
                    #mat[i][j+1] = min(mat[i][j]+1, mat[i][j+1])   
                    #delete
                    #mat[i+1][j] = min(mat[i][j]+1, mat[i+1][j])   
                else:
                    mat[i+1][j+1] = min(mat[i][j], mat[i+1][j+1])
                
                mat[i][j+1] = min(mat[i][j]+1, mat[i][j+1])
                mat[i+1][j] = min(mat[i][j]+1, mat[i+1][j])
                    
        
        
        
        '''
        for i in range(len1):
            for j in range(len2):
                if i == 0 and j == 0:
                    if word1[i] == word2[j]:
                        mat[i][j] = 0
                    else:
                        mat[i][j] = 1
                    continue
                
                if word1[i] == word2[j]:
                    if i > 0 and j >0:
                        mat[i][j] = min(mat[i-1][j-1], mat[i][j])
                    if j > 0:
                        mat[i][j] = min(mat[i][j-1], mat[i][j])
                    if i > 0:
                        mat[i][j] = min(mat[i-1][j], mat[i][j])
                else:
                    # replace prev
                    if i >0  and j>0:
                        mat[i][j] = min(mat[i-1][j-1] + 1, mat[i][j])
                    
                    # insert prev
                    if j>0:
                        mat[i][j] = min(mat[i][j-1] + 1, mat[i][j])
                    
                    # delete prev
                    if i > 0:
                        mat[i][j] = min(mat[i-1][j] + 1, mat[i][j])
        '''
        pprint.pprint(mat)
        
        return mat[len1][len2]
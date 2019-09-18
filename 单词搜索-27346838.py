class Solution:
    def check(self, board, subword, x,y, visited):
        if len(subword) == 0:
            return True
        #print(x,y,subword)
        if x > 0 and subword[0] == board[x-1][y] and (x-1,y) not in visited:
            if len(subword) == 1:
                return True
            visited.add((x-1,y))
            ret = self.check(board, subword[1:], x-1, y, visited)
            visited.remove((x-1,y))
            
            if ret:
                return True
        
        if y > 0 and subword[0] == board[x][y-1] and (x,y-1) not in visited:
            if len(subword) == 1:
                return True
            visited.add((x,y-1))
            ret = self.check(board, subword[1:], x, y-1, visited)
            visited.remove((x,y-1))
            
            if ret:
                return True
        
        if x < len(board)-1 and subword[0] == board[x+1][y] and (x+1,y) not in visited:
            if len(subword) == 1:
                return True
            visited.add((x+1,y))
            ret = self.check(board, subword[1:], x+1, y, visited)
            visited.remove((x+1,y))
            
            if ret:
                return True
        
        #print(x,y,subword)
        if y < len(board[0])-1 and subword[0] == board[x][y+1] and (x, y+1) not in visited:
            if len(subword) == 1:
                return True
            visited.add((x,y+1))
            ret=self.check(board, subword[1:], x, y+1, visited)
            visited.remove((x,y+1))
            
            if ret:
                return True
            
        return False
        
    def exist(self, board: List[List[str]], word: str) -> bool:
        x = len(board)
        y = len(board[0])
        
        c = word[0]
        
        for i in range(x):
            for j in range(y):
                if c == board[i][j] and self.check(board, word[1:], i,j, set([(i,j)])):
                    return True
        
        return False
                    
                    
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        x,y = click
        if board[x][y] == 'M':
            board[x][y] = 'X'
            return board
        max_x = len(board)
        max_y = len(board[0])
        queue = []
        if board[x][y] != 'E':
            return board
        
        queue.append((x,y))
        
        while queue:
            x,y = queue.pop(0)
            if board[x][y] == 'E':
                m_cnt = 0
                
                if x>0 and y>0:
                    if board[x-1][y-1] == 'M':
                        m_cnt +=1
                    #elif board[x-1][y-1] == 'E':
                    #    queue.append((x-1,y-1))
                if x>0:
                    if board[x-1][y] == 'M':
                        m_cnt +=1
                    #elif board[x-1][y] == 'E':
                    #    queue.append((x-1,y))
                if x>0 and y<max_y-1:
                    if board[x-1][y+1] == 'M':
                        m_cnt +=1
                    #elif board[x-1][y+1] == 'E':
                    #    queue.append((x-1,y+1))
                if x<max_x-1 and y>0:
                    if board[x+1][y-1] == 'M':
                        m_cnt += 1
                    #elif board[x+1][y-1] == 'E':
                    #    queue.append((x+1,y-1))
                if x<max_x-1:
                    if board[x+1][y] == 'M':
                        m_cnt +=1
                    #elif board[x+1][y] == 'E':
                    #    queue.append((x+1,y))
                if x<max_x-1 and y<max_y-1:
                    if board[x+1][y+1] == 'M':
                        m_cnt +=1
                    #elif board[x+1][y+1] == 'E':
                    #    queue.append((x+1,y+1))
                if y<max_y-1:
                    if board[x][y+1] == 'M':
                        m_cnt +=1
                    #elif board[x][y+1] == 'E':
                    #    queue.append((x,y+1))
                if y>0 :
                    if board[x][y-1] == 'M':
                        m_cnt +=1
                    #elif board[x][y-1] == 'E':
                    #    queue.append((x,y-1))
                
                if m_cnt > 0:
                    board[x][y] = str(m_cnt)
                else:
                    board[x][y] = 'B'
                    if x>0 and y>0 and board[x-1][y-1] == 'E':
                        queue.append((x-1,y-1))
                    if x>0 and board[x-1][y] == 'E':
                        queue.append((x-1,y))
                    if x>0 and y<max_y-1 and board[x-1][y+1] == 'E':
                        queue.append((x-1,y+1))
                    if x<max_x-1 and y>0 and board[x+1][y-1] == 'E':
                        queue.append((x+1,y-1))
                    if x<max_x-1 and board[x+1][y] == 'E':
                        queue.append((x+1,y))
                    if x<max_x-1 and y<max_y-1 and board[x+1][y+1] == 'E':
                        queue.append((x+1,y+1))
                    if y<max_y-1 and board[x][y+1] == 'E':
                            queue.append((x,y+1))
                    if y>0 and board[x][y-1] == 'E':
                        queue.append((x,y-1))
                    
        return board
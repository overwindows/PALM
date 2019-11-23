class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = [set() for _ in range(9)]
        rows = [set() for _ in range(9)]
        blocks = [set() for _ in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    n = int(board[i][j])
                    if n not in cols[j]:
                        cols[j].add(n)
                    else:
                        return False

                    if n not in rows[i]:
                        rows[i].add(n)
                    else:
                        return False
                    
                    block_r = i // 3
                    block_c = j // 3

                    block_ix = block_r * 3 + block_c

                    if n not in blocks[block_ix]:
                        blocks[block_ix].add(n)
                    else:
                        #print(blocks)
                        return False

        return True
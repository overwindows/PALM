class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def binSearchRow(row,st,ed,matrix,target):
            while st<ed:
                mid = (st+ed)//2
                if matrix[row][mid] == target:
                    return -1,-1
                elif matrix[row][mid] > target:
                    ed = mid-1
                else:
                    st = mid+1
            
            if matrix[row][st] > target:
                return st-1,st
            elif matrix[row][st] < target:
                return st,st+1
            else:
                return -1,-1
         
        def binSearchCol(col,st,ed,matrix,target):
            while st<ed:
                mid = (st+ed)//2
                if matrix[mid][col] == target:
                    return -1,-1
                elif matrix[mid][col] > target:
                    ed = mid-1
                else:
                    st = mid+1
            
            if matrix[st][col] > target:
                return st-1,st
            elif matrix[st][col] < target:
                return st,st+1
            else:
                return -1,-1
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        visited_row = [False] * len(matrix)
        visited_col = [False] * len(matrix[0])
        
        visited_row[0] = True
        visit_path = []
        st = 0
        ed = len(matrix[0])-1
        visit_path.append((0,-1,st,ed))
        
        while visit_path:
            row,col,st,ed = visit_path.pop(0)
            #print(row,col,st,ed)

            if row > -1:
                st,ed = binSearchRow(row,st,ed,matrix,target)
                #print(row,None,st,ed)
            else:
                st,ed = binSearchCol(col,st,ed,matrix,target)
                #print(None,col,st,ed)
            
            if st==-1 and ed==-1:
                return True
            
            if row > -1:
                #print(st,ed)
                if ed < len(matrix[0]) and not visited_col[ed]:
                    visit_path.append((-1,ed,0,row))
                    visited_col[ed] = True
                if st > -1 and not visited_col[st]:
                    visit_path.append((-1,st,row,len(matrix)-1))
                    #print(visit_path)
                    visited_col[st] = True
            else:
                if st > -1 and not visited_row[st] :
                    visit_path.append((st,-1,col,len(matrix[0])-1))
                    visited_row[st] = True
                if ed < len(matrix) and not visited_row[ed]:
                    visit_path.append((ed,-1,0,col))
                    visited_row[ed] = True
                    
        return False
            
            
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        rowNum = [0]*len(s)
        buf = []
        ix = 0
        for i in range(len(s)):
            if ix == 0:
                dlt = 1
            if ix == (numRows-1):
                dlt = -1
                
            rowNum[i] = ix
            if numRows > 1:
                ix += dlt
        #print(rowNum)
        
        for r in range(numRows):
            for i in range(len(s)):
                if rowNum[i] == r:
                    buf.append(s[i])
        
        return ''.join(buf)
        
        
        
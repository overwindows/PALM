class Solution:
    def myAtoi(self, str: str) -> int:
        start_pos = 0
        flag = False
        
        for _ in range(len(str)):
            if str[start_pos] == ' ':
                start_pos += 1
                continue
            else:
                break
        
        if start_pos == len(str):
            return 0
        
        if str[start_pos] == '-':
            flag = True
            start_pos += 1
        elif str[start_pos] == '+':
            flag = False
            start_pos += 1
        elif ord(str[start_pos]) > ord('9') or ord(str[start_pos]) < ord('0'):
            return 0
        
        if start_pos < len(str) and ord(str[start_pos]) <= ord('9') and ord(str[start_pos]) >= ord('0'):
            integer = ord(str[start_pos]) - ord('0')
            start_pos += 1
        else:
            return 0
        
        for i in range(start_pos, len(str)):
            if ord(str[i]) <= ord('9') and ord(str[i]) >= ord('0'):
                integer = integer * 10 + (ord(str[i]) - ord('0'))
            else:
                break
        
        if flag:
            integer = -1 * integer
            return max(integer, -2**31)
        else:
            return min(integer, 2**31-1)
        
        
        
        
            
        
        
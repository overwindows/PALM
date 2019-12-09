class Solution:
    def isPalindrome(self, s: str) -> bool:
        if not s:
            return True
        s = s.lower()
        palindrome = []
        
        for c in s:
            if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9'):
                palindrome.append(c)
        
        if len(palindrome) == 0:
            return True
        
        st = 0
        ed = len(palindrome) -1
        
        #print(palindrome)
        while st < ed:
            if palindrome[st] == palindrome[ed]:
                st += 1
                ed -= 1
            else:
                return False
        
        return True
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.fifo = []
        self.size = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            val = self.cache[key]
            self.fifo.remove(key)
            self.fifo.append(key)
            return val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            self.cache[key] = value
            self.fifo.append(key)
            if self.size == self.capacity:
                del self.cache[self.fifo.pop(0)]
            else:
                self.size += 1
        else:
            self.cache[key] = value
            self.fifo.remove(key)
            self.fifo.append(key)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
# LeetCode刷题记录

## 树

### 翻转二叉树 （剑指27） 

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

- 前序遍历，递归

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        else:
            root.left, root.right = root.right, root.left
            root.left = self.mirrorTree(root.left)
            root.right = self.mirrorTree(root.right)

            return root
```



###  二叉树的最大深度 （剑指55-I）

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

#### 回溯法（深度优先搜索DFS）

- 到达叶节点时需要进行MAX比较，选择答案
- 计数变量需要在后序位置进行退回操作
- 时间复杂度`O(N)`；空间复杂度`O(N)`

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        self.count = 0
        self.depth = 0
        def dfs(root: Optional[TreeNode]):
            if root is None:
                self.depth = max(self.depth, self.count)
                return
            
            self.count += 1
            dfs(root.left)
            dfs(root.right)
            self.count -= 1
        
        dfs(root)

        return self.depth
```



#### 动态规划（分解问题）

- 根据左右子树的最大深度推出原二叉树的最大深度
- `depth = max(maxDepth(root.left), maxDepth(root.right)) + 1`
- 时间复杂度`O(N)`；空间复杂度`O(N)`

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
          return 0
      
        depth = max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        return depth
```



#### 层序遍历（广度优先搜索BFS）

- 使用队列queue实现BFS，tmp记录该层所有叶节点，最后用tmp给queue赋值
- 每遍历一层，进行计数
- **区分`while list`和`while list is not None`**：`while list is not None`不管list是空还是有元素，list都不是None，即`list is not None`都是True
- 时间复杂度`O(N)`；空间复杂度`O(N)`

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: 
            return 0
        queue= [root]
        depth = 0
        
        while queue:
            tmp = []
            for node in queue:
                if node.left is not None:
                    tmp.append(node.left)
                if node.right is not None:
                    tmp.append(node.right)
            queue = tmp
            depth += 1
        
        return depth
```



### 平衡二叉树 （剑指55-II）

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

* 平衡二叉树定义：某二叉树中任意节点的左右子树的深度相差不超过1

* 分治思想：平衡二叉树 = 左右子树都是平衡二叉树 and 自己是平衡二叉树（由左右最大深度相差判断）
* 求二叉树最大深度：动态规划思想

```python
class Solution:
    def dfs(self, root: TreeNode):
        if root is None:
            return 0

        leftDepth = self.dfs(root.left)
        rightDepth = self.dfs(root.right)
        depth = max(leftDepth, rightDepth) + 1
        return depth

    def isBalanced(self, root: TreeNode) -> bool:
        if root is None:
            return True
        left_depth = self.dfs(root.left)
        right_depth = self.dfs(root.right)

        if abs(left_depth - right_depth) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)
```



###  二叉搜索树的第k大节点 （剑指54）

 [剑指54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

- 二叉搜索树性质：中序遍历得到升序排列
- 递归
- 时间复杂度`O(N)`；空间复杂度`O(N)`

```python
class Solution:
    def findOrder(self, root: TreeNode, array: int) -> list:
        if root is None:
            return
        self.findOrder(root.left, array)
        array.append(root.val)
        self.findOrder(root.right, array)

        return array

    def kthLargest(self, root: TreeNode, k: int) -> int:
        array = []
        if root is None:
            return None

        self.findOrder(root, array)
        return array[-k]
```



### 二叉（搜索）树的最近公共祖先 （剑指68-I, 68-II）

[235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

利用二叉搜索树的性质，while循环即可

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

- **后序遍历**
- 求最小公共祖先，需要从底向上遍历，那么二叉树，只能通过后序遍历（即：回溯）实现从低向上的遍历方式
- 在回溯的过程中，必然要遍历整棵二叉树，因为**要使用递归函数的返回值（也就是代码中的left和right）做逻辑判断**
- 时间复杂度`O(N)`；空间复杂度`O(N)`

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # base condition
        if root is None or root is p or root is q:
            return root

        # 返回值是True则代表该子树内存在p或q
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        elif not right:
            return left
        else:
            return root
```



### 重建二叉树 （剑指07）

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

[106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

* 从前序\后序中得到root节点，确定root节点在中序里的idx，继而截断 前序\后序 和 中序，进行递归
* 前序 + 后序 无法确定二叉树，因为它无法区分左右子树



### 从上到下打印二叉树 （剑指32-I, 32-II, 32-III）

[面试题32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

* 注意根节点为空的特殊情况
* 最后添加到结果列表中的是node.val，不是node

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

* BFS，队列模拟，record列表记录每层数据并加入res列表
* ==**bug**：边遍历边删除时==，不能用`for x in list`, 要用`for _ in range(len(list))`

[剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

* 使用布尔变量reverse记录是否需要倒序该层
* `list.reverse()`



### 二叉搜索树与双向链表 （剑指36）

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

* 根据二叉搜索树性质，中序遍历即为升序顺序，符合题目要求，链表搭建的操作要在中序位置
* `self.head`用来记录链表首元素；`self.pre`用来操作双向链表的搭建
* 最后注意首尾的衔接，`self.pre`最后记录的就是尾节点

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        self.head = None
        self.pre = None

        def dfs(root):
            if root is None:
                return

            dfs(root.left)
            if self.pre is None:
                self.head = root
            else:
                root.left = self.pre
                self.pre.right = root
            
            self.pre = root

            dfs(root.right)
        
        if root is None:
            return self.head

        dfs(root)
        self.pre.right = self.head 
        self.head.left = self.pre

        return self.head
```



### 二叉树中和为某值的所有路径 (剑指34)

 [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

* Bug：`res.append(list(ans))`
* 因为要求所有路径，所以需要遍历二叉树，得到目前结果就记录，且要注意回溯操作`pop`

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:

        res, ans = [], []

        def dfs(root: TreeNode, tar: int):
            if root is None:
                return 

            ans.append(root.val)

            if tar == root.val and not root.left and not root.right:
                res.append(list(ans))

            dfs(root.left, tar - root.val)
            dfs(root.right, tar - root.val)

            ans.pop(-1)
        
        dfs(root, targetSum)
        return res
```



### 对称二叉树 (剑指28)

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

* 对称二叉树的镜像就是自身: 利用判断两棵树是否为镜像的方法, 将该二叉树与自身进行对比

```python
class Solution:
    def isMirror(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if root1 is None and root2 is None:
            return True
        elif root1 is None or root2 is None:
            return False
        else:
            if root1.val == root2.val:
                return self.isMirror(root1.left, root2.right) and self.isMirror(root1.right, root2.left)
            else:
                return False

    def isSymmetric(self, root: TreeNode) -> bool:
        return self.isMirror(root, root)
```



### 合法二叉搜索树后续遍历结果 (剑指33)

[剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

* 分解问题,找到左右子树,并判断当前合法
* 合法 = 当前合法 and 左右子树合法(递归)
* 终止条件: 剩余一个节点时

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:

        def recur(i, j):
            if i >= j: return True

            p = i
            while (postorder[p] < postorder[j]): p += 1
            m = p
            while (postorder[p] > postorder[j]): p += 1

            return p == j and recur(i, m-1) and recur(m, j-1)
        
        return recur(0, len(postorder)-1)
```



### 二叉树的序列化与反序列化 (剑指37)

[297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

#### 1. 前序遍历 + DFS

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """

        if root is None:
            return '#'
        value = root.val
        return str(root.val) + ',' + self.serialize(root.left) + ',' + self.serialize(root.right)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        order = data.split(',')

        def dfs(dataList):
            val = dataList.pop(0)
            if val == '#':
                return None
            root = TreeNode(int(val))
            root.left = dfs(dataList)
            root.right = dfs(dataList)
            return root
        
        return dfs(order)
```

#### 2. 层序遍历 + BFS

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        queue = [root]
        res = []
        while queue:
            node = queue.pop(0)
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('#')
        return ','.join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        dataList = data.split(',')
        root = TreeNode(int(dataList[0]))
        queue = [root]
        i = 1
        while queue:
            node = queue.pop(0)
            if dataList[i] != '#':
                node.left = TreeNode(int(dataList[i]))
                queue.append(node.left)
            i += 1
            if dataList[i] != '#':
                node.right = TreeNode(int(dataList[i]))
                queue.append(node.right)
            i += 1
        return root
```



### 树的子结构 (剑指26)

[剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

前序遍历 + DFS

B是A的子结构 = B是A or B是A左\右子树的子结构

```python
class Solution:
    def recur(self, A: TreeNode, B: TreeNode) -> bool:
        if B is None:
            return True
        if A is None or A.val != B.val:
            return False
        
        return self.recur(A.left, B.left) and self.recur(A.right, B.right)

    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        return A is not None and B is not None and (self.recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))
```



### 二叉树中的最大路径和

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

递归

- 返回值为当前node的"贡献值": $root.val + max(left, right)$
- 其中不断更新最长路径值: $tmp = root.val + left + right$
- $maxVal$初始化为负无穷, 因为可能所有节点均为负

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxVal = float("-inf")
        def maxGain(root):
            if root is None:
                return 0
            
            left = max(0, maxGain(root.left))
            right = max(0, maxGain(root.right))
            tmp = root.val + left + right
            self.maxVal = max(self.maxVal, tmp)

            return root.val + max(left, right)
        
        maxGain(root)
        return self.maxVal
```



## 双指针与滑动窗口

### 合并两个有序链表

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

- 应用开头的**虚拟头结点**
- 双指针分别对应两个链表

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        cur = head
        while list1 is not None and list2 is not None:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
                cur = cur.next
            
            else:
                cur.next = list2
                list2 = list2.next
                cur = cur.next
        
        if list1 is None:
            cur.next = list2
        else:
            cur.next = list1

        return head.next
```



### 链表的中间结点

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

快慢指针，遍历一次链表即可，分结点个数为奇数或者偶数两种情况

注意判断条件：`while fast is not None（奇数） and fast.next is not None（偶数）`

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
```



### 调整数组顺序使奇数位于偶数前面 (剑指21)

[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

前后双指针，用两个 index 分别指向数组的前面和后面：初始情况left=0, right=len-1，不占用额外内存       

我们希望left都指向奇数，right都指向偶数        

- 如果 left 是偶数，right是奇数：交换        
- 如果 left 是奇数：left++        
- 如果 right 是偶数：right--



### 链表中倒数第k个结点 (剑指22)

[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

双指针，快指针先行，固定距离，返回慢指针即可



### 删除链表的倒数第 N 个结点

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

- 基本思路与“链表中倒数第k个节点”一致
- **==需要设置dummy虚拟头节点！！！防止删除头节点的时候引起bug==**

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        fast, slow = dummy, dummy

        for i in range(n):
            fast = fast.next
        
        while fast.next is not None:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return dummy.next
```



### NSum问题

#### 和为s的数字 (剑指57， 剑指57-II)

[剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

升序数组，双指针

[剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

双指针，移动计算区间加和

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        left, right = 1, 2
        res = []
        while right <= target // 2 + 1:
            total_sum = (left + right) * (right - left + 1) / 2
            if  total_sum == target:
                res.append(list(range(left, right+1)))
                left += 1 # 注意这一步，每得到一个结果都需要移动左指针或右指针，否则会进入死循环
            elif total_sum < target:
                right += 1
            else:
                left += 1

        return res 
```



#### 三数之和

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

重点是去除重复的答案

**每次移动k, i, j的时候都要主动排除重复值 and 数组不越界**

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []

        for k in range(0, len(nums)-2):
            if nums[k] > 0 :
                break
            if k > 0 and nums[k] == nums[k-1]:
                continue
            
            i, j = k+1, len(nums)-1
            while i < j:
                total = nums[k] + nums[i] + nums[j]
                if total == 0:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i-1]: i += 1
                    while i < j and nums[j] == nums[j+1]: j -= 1
                elif total < 0:
                    i += 1
                    while i < j and nums[i] == nums[i-1]: i += 1
                else:
                    j -= 1
                    while i < j and nums[j] == nums[j+1]: j -= 1

        return res
```





### 颠倒字符串中的单词 (剑指58-I， 剑指58-II)

[151. 颠倒字符串中的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

思路：双指针，初始化指向字符串尾部

- 前指针找到单词开头，得到单词
- 前指针跳过空格部分，指向上一单词最后一个字符
- 后指针=前指针，继续循环

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        res = []
        i, j = len(s) - 1, len(s) - 1
        while i >= 0:
            while s[i] != ' ' and i >= 0: # 注意最后结束循环的情况（i= -1）
                    i -= 1
                
            res.append(s[i+1:j+1])

            while s[i] == ' ':
                i -= 1
            j = i
        
        return " ".join(res)
```



[剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

很简单，切片或者遍历



### 相交链表（剑指52）

使得两指针相遇的地点为公共节点

[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            if A is None: A = headB
            else: A = A.next

            if B is None: B = headA
            else: B = B.next

        return A
```



### 无重复字符的最长子串 （剑指48）

[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

- 滑动窗口-双指针
- 右指针指向重复的值时，左指针负责向右移动将其移出（通过左指针原位置与哈希表记录的比较）
- max操作求存在的最大长度

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        dict = {}
        res = 0
        i = -1

        for j in range(len(s)):
            if s[j] in dict:
                i = max(dict[s[j]], i) # 易错点
            dict[s[j]] = j
            res = max(res, j - i)
        
        return res
```



### 环形链表

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

- 应用快慢指针，如果有环的话，快指针会超圈赶上慢指针
- 如果快指针到达链表结尾则证明没有环形

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast, slow = head, head
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False
```



[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

- 在141的基础之上，在快慢指针第一次相遇之后（此时快比慢多走了k步），将快指针指向head
- 之后快慢指针同步伐前进，再次相遇的点即为环起点

<img src="D:\TJU\实习\LeetCode刷题记录\image-20220324150413402-16487089436281.png" alt="image-20220324150413402" style="zoom:50%;" />

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        if fast is None or fast.next is None:
            return None

        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return fast
```



### 反转链表

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur != None:
            next_node = cur.next
            cur.next = pre
            pre = cur
            cur = next_node
```



[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

```python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        for _ in range(left-1):
            pre = pre.next
        cur = pre.next
        for _ in range(right - left):
            next_node = cur.next
            cur.next = next_node.next
            next_node.next = pre.next
            pre.next = next_node

        return dummy.next
```





### K 个一组翻转链表

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

递归求解, 上一分段的right, 当作反转操作中的pre

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1:
            return head
        
        left, right = head, head

        for _ in range(k-1):
            # 注意链表最后结尾不足的情况
            if right is None or right.next is None:
                return head
            right = right.next
        
        # 将下一阶段的链表right节点作为链表反转的pre节点!!!
        pre = self.reverseKGroup(right.next, k)

        # 常规链表反转操作
        for _ in range(k):
            nextNode = left.next
            left.next = pre
            pre = left
            left = nextNode
        
        return right
```



### 重排链表

[143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

分三步走:
找到中点, 反转右部链表, 合并左右链表

```python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        分三步走:
        找到中点, 反转右部链表, 合并左右链表
        """
        mid = self.findMid(head)
        p = mid.next
        mid.next = None
        tmp = self.reverse(p)
        self.merge(head, tmp)
    
    def findMid(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow  = slow.next
        return slow
    
    def reverse(self, head: ListNode) -> ListNode:
        preNode, curNode = None, head
        while curNode is not None:
            nextNode = curNode.next
            curNode.next = preNode
            preNode = curNode
            curNode = nextNode
        return preNode
    
    def merge(self, l1: ListNode, l2: ListNode):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l2.next = l1_tmp

            l1 = l1_tmp
            l2 = l2_tmp
```



## 优先队列 (3)

### 合并K个升序链表

[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

- 应用最小堆(优先队列)，首先将所有链表的头节点加入最小堆，之后的思路和合并两个的类似
- **注意：ListNode需要定义比较运算符！！否则无法加入最小堆**

​		`ListNode.__lt__ = __lt__`       `lt`含义为 less than

- 时间复杂度：$N*logK$
- 空间复杂度：$O(K)$

```python
from heapq import *
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def __lt__(self,other):
            return self.val < other.val
        ListNode.__lt__ = __lt__

        small = []
        head = ListNode()
        cur = head

        for i in range(len(lists)):
            if lists[i] is not None:
                heappush(small, lists[i])
        
        while len(small) != 0:
            # 合并操作
            tmp = heappop(small)
            cur.next = tmp
            cur = cur.next
            # 更新最小堆
            if tmp.next is not None:
                heappush(small, tmp.next)
        
        return head.next
```



### 滑动窗口的最大值 (剑指59-I)

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

**解法一：优先队列**

- 先将前k个元素和其index加入最小堆，然后循环模拟窗口滑动
- 如果当前堆顶不在窗口内，就将其pop，直至堆顶元素在窗口内，将其添加入res列表
- 时间复杂度为$O(nlogn)$；空间复杂度为$O(n)$

```python
import heapq
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if  k == 0:
            return []
            
        # 优先队列
        q = [(-1 * nums[i], i) for i in range(k)]
        heapq.heapify(q)
        res = [-1 * q[0][0]]

        for i in range(k, len(nums)):
            heappush(q, (-1 * nums[i], i))
            while q[0][1] <= i - k:
                heappop(q)
            res.append(-1 * q[0][0])
        
        return res
```

**解法二：单调队列**

- 单调队列维护单调递减性策略：每次加入新元素的时候，pop队末所有小于新元素的元素
- 根据本题内容，需要在单调队列维护操作的基础上增加对于最大值出窗的判断
- 时间复杂度为$O(n)$，每一个下标恰好被放入队列一次，并且最多被弹出队列一次
- 空间复杂度为$O(k)$

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = []
        res = []

        for i, j in zip(range(1 - k, len(nums) + 1 - k), range(0, len(nums))):

            if i > 0 and queue[0] == nums[i - 1]: # 若最大值离开窗口
                queue.pop(0)
            while queue and queue[-1] < nums[j]: # 单调递减，若队末元素<新加入元素则pop
                queue.pop(-1)
            queue.append(nums[j])
            if i >= 0:
                res.append(queue[0])
        
        return res
```



### 数据流的中位数 (剑指41)

[295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

**优先队列，小顶堆+大顶堆**

**主要思想：总使得`小顶堆内元素 <= 大顶堆内元素 + 1`， 则中位数则为小顶堆栈顶元素或者两个栈顶的平均**

- 如果当前元素为奇数个，那么就先将num加入小顶堆，然后从小顶堆pop一个元素到大顶堆
- 如果当前元素为偶数个，那么就先将num加入大顶堆，然后从大顶堆pop一个元素到小顶堆
- 注意大顶堆保存的时候要取负值，以保证大顶堆性质

> **python里通常创建的是最小堆**, 对于最小堆来说, 堆中索引位置0即heap[0]的值一定是堆中的最小值, 并且堆中的每个元素都符合公式$heap[k] <= heap[k*2+1]和 heap[k] <= heap[k*2+2]$, 其中heap[k]是父节点, 而heap[k*2+1]和heap[k*2+2]是heap[k]的子节点, 父节点永远小于等于它自己的子节点

```python
from heapq import *

class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        if len(self.small) == 0:
            heappush(self.small, num)
        elif len(self.small) == len(self.large):
            heappush(self.large, -1 * num)
            heappush(self.small, -1 * heappop(self.large))
        else:
            heappush(self.small, num)
            heappush(self.large, -1 * heappop(self.small))

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (self.small[0] - self.large[0]) / 2
        else:
            return self.small[0]
```



## 排序算法

**[七种排序总结文章](https://leetcode-cn.com/problems/sort-an-array/solution/dong-hua-mo-ni-yi-ge-kuai-su-pai-xu-wo-x-7n7g/)**

### 归并排序

分治递归思想, 时间复杂度为$O(nlogn)$,$logn$层递归,每层还需要$n$次比较操作

对比操作分类讨论: (代码中可以两两合并)

- 情况一: 左空, 放右
- 情况二: 右空, 放左
- 情况三: 非空 and 左<=右, 放左
- 情况四: 非空 and 左>右, 放右

[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge_sort(left, right):
            if left >= right:
                return

            mid = (left + right) // 2
            merge_sort(left, mid)
            merge_sort(mid + 1, right)

            i = left
            j = mid + 1
            tmp[left:right + 1] = nums[left:right + 1]
            for k in range(left, right + 1):
                if i == mid + 1 or (j <= right and tmp[i] > tmp[j]):
                    nums[k] = tmp[j]
                    j += 1
                else:
                    nums[k] = tmp[i]
                    i += 1
        
        tmp = [0] * len(nums)
        merge_sort(0, len(nums) - 1)
        return nums
```



### 归并排序与逆序对统计 (剑指51)

[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        # 在归并排序中完成对逆序对的统计
        def merge_sort(left, right) -> int:
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            res = merge_sort(left, mid) + merge_sort(mid+1, right)

            i = left
            j = mid + 1
            tmp[left:right+1] = nums[left:right+1]
            # 情况一: 左空, 放右
            # 情况二: 右空, 放左
            # 情况三: 左<=右, 放左(情况二和情况三可以合并)
            # 情况四: 左>右, 放右,增加逆序数(m - i + 1)
            for k in range(left, right + 1):
                if i == mid + 1:
                    nums[k] = tmp[j]
                    j += 1
                elif j == right + 1 or tmp[i] <= tmp[j]:
                    nums[k] = tmp[i]
                    i += 1
                else:
                    nums[k] = tmp[j]
                    j += 1
                    res += mid - i + 1
            return res
        
        tmp = [0] * len(nums)
        return merge_sort(0, len(nums) - 1)
```



### 快速排序

时间复杂度为$O(logN)$

==重点为双指针元素交换==

```python
i, j = l - 1, l
        for j in range(l, r):
            if nums[j] < pivot:
                i += 1
                nums[i], nums[j]= nums[j], nums[i]
        
        nums[i + 1], nums[r] = nums[r], nums[i + 1]
```

完整流程(三个函数):

```python
class Solution:
    def random_partition(self, nums: List[int], l: int, r: int) -> List[int]:
        pivot_idx = random.randint(l, r)
        pivot = nums[pivot_idx]
        nums[r], nums[pivot_idx] = nums[pivot_idx], nums[r]
        i, j = l - 1, l
        for j in range(l, r):
            if nums[j] < pivot:
                i += 1
                nums[i], nums[j]= nums[j], nums[i]
        
        nums[i + 1], nums[r] = nums[r], nums[i + 1]
        return i + 1           

    def quick_sort(self, nums: List[int], l: int, r: int):
        if l >= r:
            return
        mid = self.random_partition(nums, l, r)
        self.quick_sort(nums, l, mid - 1)
        self.quick_sort(nums, mid + 1, r)
    
    def sortArray(self, nums: List[int]) -> List[int]:
        self.quick_sort(nums, 0, len(nums) - 1)
        return nums
```



### 快速选择算法 (解决TopK问题)

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

- 第K个元素, 与前后子序列的顺序都无关, 只需要将pivot_idx和目标K进行比较, 等于就返回nums[pivot_idx]
- 否则直接只在左序列或者右序列其一继续寻找符合目标的pivot即可
- 时间复杂度为$O(N)$

```python
class Solution:
    def random_partition(self, nums: List[int], l: int, r:int) -> int:
        p_idx= random.randint(l, r)
        pivot = nums[p_idx]
        nums[r], nums[p_idx] = nums[p_idx], nums[r]
        i, j = l-1, l 
        for j in range(l, r):
            if nums[j] > pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1], nums[r] = nums[r], nums[i + 1]
        return i + 1

    def findKthLargest(self, nums: List[int], k: int) -> int:
        l, r = 0, len(nums) - 1
        while True:
            mid = self.random_partition(nums, l, r)
            if mid == k - 1:
                return nums[mid]
            elif mid < k - 1:
                l = mid + 1
            else:
                r = mid - 1
```





## 二分查找

### 直接二分查找

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        i = 0
        j = len(nums) - 1

        while i <= j:
            mid = (i + j) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                j = mid - 1
            else:
                i = mid + 1
        
        return -1
```



### 查找元素的第一个和最后一个位置 (剑指53-I)

**在具体的算法问题中，常用到的是「搜索左侧边界」和「搜索右侧边界」这两种场景**

==**「搜索左侧边界」**当找到 target 时，收缩右侧边界==

==**「搜索右侧边界」**当找到 target 时，收缩左侧边界==

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

查找target的左右边界 等价于 查找target的右边界 and 查找target-1的右边界 + 1

```python
class Solution:
    def helper(self, nums: List[int], target: int) -> List[int]:
        # 二分查找target右边界
        i = 0
        j = len(nums) - 1
        while i <= j:
            mid = (i + j) // 2
            if nums[mid] <= target:
                i = mid + 1
            else:
                j = mid - 1
        return j

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res = [-1, -1]
        if target in nums:
            res[0] = self.helper(nums, target - 1) + 1
            res[1] = self.helper(nums, target) 
        return res
```



###  0～n-1中缺失的数字 (剑指53-II)

[剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

或者可以直接用数学法

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if mid < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return left
```



### 二维矩阵中的查找 (剑指04)

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

从矩阵右上角或者左下角开始搜索

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0:
            return False
        # 从右上开始搜索
        i = 0
        j = len(matrix[0]) - 1
        while i < len(matrix) and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
```



### 旋转排序数组最小值

==比较重要,需要记忆==

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/) (无重复) 

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/) (有重复)  (剑指11)

```python
def findMin(self, nums: List[int]) -> int:
        '''
        如果左端点的值小于右端点的值则可以提前退出。
        否则我们选取中点，并判断中点的位置是在左边有序部分还是右边有序部分。
        如果在左边有序部分，那么 r = mid，如果在右边有序部分则 l = mid + 1
        **特殊情况：numbers[mid] == numbers[right]
        选择舍弃右端点，因为舍弃右端点不会错过最小值
        一个前提：取中点逻辑是向下取整，如果你取中点是向上取整情况就有所不同了。
        '''
        left = 0
        right = len(nums) - 1

        while left != right :
            # 如果左端点的值小于右端点的值则可以提前退出
            if nums[left] < nums[right] :
                return nums[left]
            
            mid = (left + right) // 2

            # 比较右端点而不是左端点
            # eg. [2,2,2,0,1] 若比较左端点会错过min值
            if nums[mid] < nums[right] :
                right = mid
            elif nums[mid] > nums[right] :
                left = mid + 1
            else:
                right -= 1
        
        return nums[left]
```



### 搜索旋转排序数组

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

- 最简单方法：在153基础上，将问题转换为升序数组二分查找
- **提升方法：将数组一分为二，其中一定有一个是有序的，另一个可能是有序，也能是部分有序。此时有序部分用二分法查找。无序部分再一分为二，其中一个一定有序，另一个可能有序，可能无序。就这样循环.**

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 左数组有序
            elif nums[left] <= nums[mid]:
                if nums[left] <= target and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # 右数组有序
            else:
                if nums[mid] < target and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```



### 两个正序数组的中位数

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

转化问题: 寻找两个有序数组中第K大的数

每次排除 2/k个数, 时间复杂度$O(M+N)$

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 寻找第K大的数
        def getKthElement(k):
            index1, index2 = 0, 0 # 两个变量记录不断缩小的查找范围
            while True:
                # 特殊情况
                if index1 == len(nums1): # nums1已完全排除
                    return nums2[index2 + k - 1]
                if index2 == len(nums2): # nums2已完全排除
                    return nums1[index1 + k - 1]
                if k == 1: # 只需返回第一个大的数, min操作即可
                    return min(nums1[index1], nums2[index2])

                # 正常情况
                # k1, k2是需要讨论部分的截点位, 数组本身长度不足就取全部
                k1 = min(index1+ k//2 - 1, len(nums1) - 1)
                k2 = min(index2 + k//2 - 1, len(nums2) - 1)
                #
                if nums1[k1] < nums2[k2]: # 舍弃nums1的前k//2个数
                    k -= k1 - index1 + 1
                    index1 = k1 + 1
                else: # 舍弃nums2的前k//2个数
                    k -= k2 - index2 + 1
                    index2 = k2 + 1
        
        m, n = len(nums1), len(nums2)
        total = m + n
        if total % 2 == 0:
            return (getKthElement(total//2) + getKthElement(total//2 + 1)) / 2
        else:
            return getKthElement(total//2 + 1)
```



### x的平方根

[69. x 的平方根 ](https://leetcode-cn.com/problems/sqrtx/)

由于 x 平方根的整数部分 $\textit{ans}$ 是满足 $k^2\leq x$ 的最大值，因此我们可以对 k 进行二分查找，从而得到答案。

不断更新res, 直到结束二分查找

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left, right = 0, x
        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans
```



## 动态规划

==**明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义**==



### 斐波那契数列 / 台阶问题 （剑指10-I，10-II）

[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```python
class Solution:
    def fib(self, n: int) -> int:
        if n == 0 or n == 1:
            return n

        dp = [0] * (n + 1) 
        dp[0], dp[1] = 0, 1
        for i in range(2, n + 1):
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007
        
        return dp[n]
```

- 每次状态转移只需要 DP table 中的一部分，只记录必要的数据，从而降低空间复杂度：

```python
class Solution:
    def fib(self, n: int) -> int:
        if n == 0 or n == 1:
            return n

        dp_2, dp_1 = 0, 1
        for i in range(2, n + 1):
            dp = (dp_1 + dp_2) % 1000000007
            dp_2 = dp_1
            dp_1 = dp
        
        return dp
```

[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)



### 整数拆分 / 剪绳子 （剑指14-I，14-II）

[343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

状态转移方程：`dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j])) `

==总共i长度，剪下j长度，里面一层max取剩余部分剪和不剪，外面一层取剪多少==

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 1

        for i in range(3, n+1):
            for j in range(2, i):
                dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j])) 
        return dp[n]
```



### 礼物的最大价值 （剑指47）

[剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:

        for i in range(1, len(grid)):
            grid[i][0] = grid[i][0] + grid[i-1][0]
        
        for j in range(1, len(grid[0])):
            grid[0][j] = grid[0][j] + grid[0][j-1]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                grid[i][j] = grid[i][j] + max(grid[i][j-1], grid[i-1][j])
        
        return grid[len(grid)-1][len(grid[0])-1]
```



### 丑数 （剑指49）

[264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

- 三指针
- ==重点：==`num[i] = min(num[a] * 2, num[b] * 3, num[c] * 5)`

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        a, b, c = 1, 1, 1
        num = [1] * (n + 1)
        for i in range(2, n + 1):
            num[i] = min(num[a] * 2, num[b] * 3, num[c] * 5)
            if num[i] == num[a] * 2:
                a += 1
            if num[i] == num[b] * 3:
                b += 1
            if num[i] == num[c] * 5:
                c += 1
        
        return num[n]
```



### 连续子数组最大和 （剑指42）

[53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

- ==dp[i] - 表示序列末端到当前位置 i 的最大子序列和==
- 注意最后需要求dp的`Max`得到结果，因为dp的意义中确定了子序列的末端

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        '''
        动态规划（五星）
        dp[i] - 表示序列末端到当前位置 i 的最大子序列和
        状态转移方程为： dp[i] = max(dp[i - 1] + nums[i], nums[i])
        '''   
        dp, res = nums[0], nums[0]

        for i in range(1, len(nums)):
            dp = max(dp + nums[i], nums[i])
            res = max(res, dp)
        
        return res
```



### 乘积最大子数组

[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

因为乘法有正负号的问题, 需要同时记录当前最大和最小, 如果当前乘数小于0, 则需要交换imax和imin

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        imax, imin, maxVal = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            if nums[i] < 0:
                imax, imin = imin, imax
            imax = max(imax*nums[i], nums[i])
            imin = min(imin*nums[i], nums[i])
            maxVal = max(imax, maxVal)

        return maxVal
```



### 股票买卖系列问题（待完）

https://labuladong.gitee.io/algo/3/27/96/

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) （剑指63）

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 最大利润 = max(前一天的最大利润， 当天的价格 - 历史最低价格)
        cost = float("inf")
        profit = 0

        for price in prices:
            cost = min(cost, price)
            profit = max(profit, price - cost)
        
        return profit
```



[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

- 在121的基础之上, 需要两个状态来进行动态规划: dp_0, dp_1

==当前无股票 = max(上一时刻无股票, 上一时刻有股票 + 当前卖出)==

==当前有股票 = max(上一时刻有股票, 上一时刻无股票 - 当前买进)==

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp_0, dp_1 = 0, -1 * prices[0]
        for i in range(1, len(prices)):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, dp_0 - prices[i])
        
        return max(dp_0, dp_1)
```



[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        由于我们最多可以完成两笔交易，因此在任意一天结束之后，我们会处于以下五个状态中的一种：
        1. 未进行过任何操作；
        2. 只进行过一次买操作；
        3. 进行了一次买操作和一次卖操作，即完成了一笔交易；
        4. 在完成了一笔交易的前提下，进行了第二次买操作；
        5. 完成了全部两笔交易。
        情况一利润为0,不需要讨论,用四个变量记录剩余四种情况的变化
        注意:允许同一天进行多项操作
        因为最终结果在max(0, sell1, sell2)中选择,且sell2一定>=sell1,所以返回sell1即可
        '''
        buy1, buy2 = -prices[0]
        sell1 = sell2  = 0
        for i in range(1, len(prices)):
            buy1 = max(buy1, -1 * prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        
        return sell2
```



### n个骰子的点数 （剑指60）需要加深理解

[剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [1/6] * 6
        for m in range(2, n+1):
            tmp = [0] * (5 * m + 1)
            for i in range(len(dp)):
                for j in range(6):
                    tmp[i + j] += dp[i] / 6
            dp = tmp
        
        return dp
```



### 把数字翻译成字符串 （剑指46）

[剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

数字范围在10-25之间的数字都有两种翻译方法

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        dp = [0] * len(s)
        dp[0] = 1
        for i in range(1, len(s)):
            tmp = 10 * int(s[i-1]) + int(s[i])
            if tmp >= 10 and tmp <= 25:
                if i == 1: # dp[1]特殊情况
                    dp[i] = 2
                else:
                    dp[i] = dp[i-1] + dp[i-2]
            else:
                dp[i] = dp[i-1]
        return dp[-1]
```



### 数字1的个数 （剑指43 hard）

[233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

- 按照位数，类似密码锁，逐位讨论
- 每位都可以分为三种情况：等于0，等于1，2-9；然后分别计算可能性

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        low = 0
        high = n // 10
        cur = n % 10
        digit = 1
        res = 0
        while high != 0 or cur != 0:
            if cur == 0:
                res += high * digit
            elif cur == 1:
                res += high * digit + low + 1
            else:
                res += (high + 1) * digit
            # 更新
            low = low + cur * digit
            cur = high % 10
            high = high // 10
            digit = digit * 10
        return res
```



### 正则表达式匹配 （剑指19 hard 梭哈梭哈）

[剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)



### 接雨水

[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        动态规划,按列计算
        当前列能接的雨水 = min(左边最高, 右边最高) - 当前列高度
        其中记录左/右最高可以使用动态规划: maxleft[i] = max(maxleft[i-1], cur_height) 
        '''
        maxleft, maxright = [0] * len(height), [0] * len(height) 
        maxleft[0], maxright[-1] = height[0], height[-1]
        for i in range(1, len(height)):
            maxleft[i] = max(maxleft[i-1], height[i])
        
        for i in range(len(height) - 2, -1, -1):
            maxright[i] = max(maxright[i+1], height[i])
        
        res = 0
        for i in range(1, len(height)-1):
            res += min(maxleft[i], maxright[i]) - height[i]
        
        return res
```



### 打家劫舍

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

- 抢和不枪两种选择: `dp[i] = max(nums[i] + dp[i-2], dp[i-1])`
- 第一家和第二家的情况需要单独讨论

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        if len(nums) == 1:
            return dp[0]
        dp[1] = max(nums[0], nums[1])
        if len(nums) == 2:
            return dp[1]

        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        
        return dp[-1]
```



### 最长回文子串

[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

==**二维数组初始化!!!**==

`dp = [[False]*len(s) for _ in range(len(s))]` 不是 `dp = [[False]*len(s)] *len(s)`

动态规划思想, 将边界条件和状态转移梳理清楚, 然后每次记录最长max_len和子串左端即可

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        max_len = 0
        dp = [[False]*len(s) for _ in range(len(s))]
        for j in range(len(s)):
            for i in range(j + 1):
                # 边界条件
                if j < i + 2:
                    if s[i] == s[j]:
                        dp[i][j] = True                       

                # 状态转移
                else:
                    if dp[i+1][j-1] == True and s[i] == s[j]:
                        dp[i][j] = True
                
                cur_len = j - i + 1
                # 更新记录
                if dp[i][j] == True and max_len < cur_len:
                    max_len = cur_len
                    start = i
                    end = j

        return s[start:end+1]
```



### 编辑距离

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

dp[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数

所以，

当 `word1[i] == word2[j]`，``dp[i][j] = dp[i-1][j-1]``；

当 `word1[i] != word2[j]`，`dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1`

其中，``dp[i-1][j-1] `表示替换情况，`dp[i-1][j] `表示删除情况，`dp[i][j-1] `表示插入情况。

注意，针对第一行，第一列要单独考虑! 单独初始化

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n+1) for _ in range(m+1)]

        if m == 0 and n == 0:
            return 0

        # 初始化
        for i in range(1, m+1):
            dp[i][0] = dp[i-1][0] + 1
        for j in range(1, n+1):
            dp[0][j] = dp[0][j-1] + 1
        
        # 更新dp数组
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        return dp[m][n]
```



### 最长公共子序列

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

二维DP

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0]* (len(text2)+1) for _ in range(len(text1)+1)]
        for i in range(1, len(text1)+1):
            for j in range(1, len(text2)+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```



### 最长递增子序列(LIS)

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

1. $O(N^2)$解法

$dp[i]$ 的值代表 `nums` 以 $nums[i]$结尾的最长子序列长度

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for j in range(1, len(nums)):
            for i in range(0, j):
                if nums[j] > nums[i]:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return max(dp)
```

2. $O(N*logN)$解法

考虑维护一个列表 $tails$，其中每个元素 $tails[k]$ 的值代表 **长度为 k+1 的子序列尾部元素的值**。

**tails列表一定是严格递增的：** 即当尽可能使每个子序列尾部元素值最小的前提下，子序列越长，其序列尾部元素值一定更大

**找到以nums[k]为基准, tails中的大小分界线, 即第一个大于nums[k]的位置**

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = [0] * len(nums)
        res = 0
        for k in range(len(nums)):
            i, j = 0, res
            while i < j:
                mid = (i + j) // 2
                if tails[mid] < nums[k]:
                    i = mid + 1
                else:
                    j = mid
            tails[i] = nums[k]
            if j == res:
                res += 1
        return res
```



[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

- 二维矩阵最优解 -> 动态规划
- 直接在原数组上操作, 不需要额外空间

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] = grid[i][j-1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i-1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i][j-1], grid[i-1][j]) + grid[i][j]
        return grid[-1][-1]
```



### 最大正方形

[221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

状态转移方程:

$dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1$

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        maxSide = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    maxSide = max(maxSide, dp[i][j])
        
        return maxSide * maxSide

```



### 单词拆分

[139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

动态规划

$dp[i] == True \ and \ s[i:j] \ in \ wordDict$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if dp[i] == True and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]
```



## 搜索问题(DFS, BFS)

### 机器人运动范围 (剑指13)

[剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

- DFS + 剪枝

```python
class Solution:
    def digitSum(self, n: int) -> int:
        res = 0
        while n != 0:
            res += n % 10
            n = n // 10
        return res

    def movingCount(self, m: int, n: int, k: int) -> int:
        visited = []
        def dfs(i: int, j: int):
            if i >= m or j >= n or (i, j) in visited or self.digitSum(i) + self.digitSum(j) > k:
                return 0
            
            visited.append((i, j))
            return 1 + dfs(i + 1, j) + dfs(i, j + 1)
        
        return dfs(0, 0)
```

- BFS + 剪枝

```python
class Solution:
    def digitSum(self, n: int) -> int:
        res = 0
        while n != 0:
            res += n % 10
            n = n // 10
        return res

    def movingCount(self, m: int, n: int, k: int) -> int:
        queue = [(0, 0)]
        count = 0
        visited = []
        while len(queue) != 0:
            pos = queue.pop(0)
            i, j = pos[0], pos[1]
            if self.digitSum(i) + self.digitSum(j) > k or (i, j) in visited:
                continue

            count += 1
            visited.append(pos)

            if i < m - 1:
                queue.append((i + 1, j))
            if j < n - 1:
                queue.append((i, j + 1))
        return count
```



### 岛屿系列问题

[**一文秒杀所有岛屿题目**](https://labuladong.gitee.io/algo/4/30/109/)

[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

- **为什么每次遇到岛屿，都要用 DFS 算法把岛屿「淹了」, 避免维护 `visited` 数组**。

- 时间复杂度：O(MN)，其中 M 和 N 分别为行数和列数。

  空间复杂度：O(MN)，在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 MN。

  

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # DFS(FloodFill 算法), 用来淹没整个岛屿
        def dfs(i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                return
            if grid[i][j] == "0":
                return
            
            grid[i][j] = "0"
            dfs(i-1, j)
            dfs(i+1, j)
            dfs(i, j+1)
            dfs(i, j-1)

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    res += 1
                    dfs(i, j)
        return res
```



[695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >=len(grid[0]):
                return 0
            
            if grid[i][j] == 1:
                grid[i][j] = 0
                return 1 + dfs(i-1, j) + dfs(i, j-1) + dfs(i+1, j) + dfs(i, j+1)
            else:
                return 0

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res = max(dfs(i, j), res)
        return res
```



### 螺旋矩阵

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

- 使用四个方向的遍历
- 注意每一层后两个遍历需要额外的苛刻的判断条件, 否则答案中会出现重复

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        if not matrix:
            return res
        left, right, top, bottom = 0, len(matrix[0])-1, 0, len(matrix)-1
        while left <= right and top <= bottom:
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            for i in range(top + 1, bottom + 1):
                res.append(matrix[i][right])
            if bottom > top:
                for i in range(right-1, left, -1):
                    res.append(matrix[bottom][i])
            if left < right:
                for i in range(bottom, top, -1):
                    res.append(matrix[i][left])
            left += 1
            right -= 1
            top += 1
            bottom -= 1
```



## 回溯

### 全排列

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def traceback(first):
            if first == n:
                res.append(nums[:]) 
                # 此处注意, 需要nums的副本nums[:], 否则答案会随nums变化
                return
            for i in range(first, n):
                nums[i], nums[first] = nums[first], nums[i]
                traceback(first + 1)
                nums[i], nums[first] = nums[first], nums[i]
        
        res = []
        n = len(nums)
        traceback(0)
        return res
```



## 字符串

### 字符串相加/大数加法

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        if num1 == "0" and num2 == "0":
            return "0"

        n1 = list(num1)
        n2 = list(num2)
        n1.reverse()
        n2.reverse()
        #补零
        if len(n1) < len(n2):
            tmp = ['0'] * (len(n2)-len(n1))
            n1.extend(tmp)
        else:
            tmp = ['0'] * (len(n1)-len(n2))
            n2.extend(tmp)
        
        # 循环进位, res初始化要多一位,考虑最高位进位
        # 进位值直接存在res列表里
        res = [0] * (len(n1) + 1)
        for i in range(len(n1)):
            add = int(n1[i]) + int (n2[i]) + res[i]
            if add < 10:
                res[i] = add
            else:
                res[i] = add % 10
                res[i+1] = 1

        # 从最高不为0的一位开始输出
        res.reverse()
        for j in range(len(res)):
            if res[j] != 0:
                return "".join([str(x) for x in res[j:]])
```





## 栈



### 有效的括号 (字符匹配)

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        self.stack = []
        def operation(s):
            for i in range(len(s)):
                c2 = s[i]
                if len(self.stack) != 0:
                    c1 = self.stack[-1]
                    if ((c1 == '(' and c2 == ')') or (c1 == '[' and c2 == ']') or (c1 == '{' and c2 == '}')):
                        self.stack.pop(-1)
                    else:
                        self.stack.append(c2)
                else:
                    self.stack.append(c2)

        operation(s)
        if len(self.stack) == 0:
            return True
        else:
            return False
```



## 设计 / 模拟

### LRU缓存

[146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)

- 双向链表 + 哈希
- 四个辅助函数

```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.dict = {}
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size =0


    def get(self, key: int) -> int:
        # 查看哈希表是否有记录
        # 情况1: 无记录返回-1
        if key not in self.dict:
            return -1

        # 情况2: 有记录 将其移动至链表头
        node = self.dict[key]
        self.move_to_head(node)
        # 返回取值
        return node.value


    def put(self, key: int, value: int) -> None:

        # 情况1: 如果哈希表无记录:
        if key not in self.dict:
            # 把新node放入表头
            node = DLinkedNode(key, value)
            self.add_to_head(node)
            # 信息存入哈希表
            self.dict[key] = node
            # 判断容量, 若超出capacity,则删除表尾, 同时删除哈希记录, size-1
            self.size += 1
            if self.size > self.capacity:
                removed = self.remove_from_tail()
                self.dict.pop(removed.key)
                self.size -= 1

        # 情况2: 如果哈希表有记录
        else: 
            # 从哈希表获得node信息
            node = self.dict[key]
            # 修改node的val
            node.value = value
            # 将node放置于表头
            self.move_to_head(node)
    

    def remove_node(self, node: DLinkedNode) -> None:
        node.pre.next = node.next
        node.next.pre = node.pre
    
    def add_to_head(self, node: DLinkedNode) -> None:
        node.next = self.head.next
        node.next.pre = node
        self.head.next = node
        node.pre = self.head
    
    def remove_from_tail(self):
        node = self.tail.pre
        self.remove_node(node)
        return node
    
    def move_to_head(self, node: DLinkedNode) -> None:
        self.remove_node(node)
        self.add_to_head(node)
```



## 找规律 恶心心

[剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)


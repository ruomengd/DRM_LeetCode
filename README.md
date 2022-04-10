# 目录



## 二叉树 (18)

- 时间复杂度: 可以理解为遍历次数
- 空间复杂度: 递归时系统需使用的栈空间大小, 即二叉树的高度

| 题目ID           | 题目名称                                                     | 时间复杂度  | 空间复杂度 |
| ---------------- | ------------------------------------------------------------ | ----------- | ---------- |
| 226 / 剑指27     | [翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/) | $O(N)$      | $O(N)$     |
| 101 / 剑指28     | [对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/) | $O(N)$      | $O(N)$     |
| 104 / 剑指55-I   | [二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) | $O(N)$      | $O(N)$     |
| 110 / 剑指55-II  | [平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/) | $O(N*logN)$ | $O(N)$     |
| 剑指54           | [二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/) | $O(N)$      | $O(N)$     |
| 235 / 剑指68-I   | [二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | $O(N)$      | $O(1)$     |
| 236 / 剑指68-II  | [二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) | $O(N)$      | $O(N)$     |
| 105 / 剑指07     | [前序与中序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | $O(N)$      | $O(N)$     |
| 106              | [中序与后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) | $O(N)$      | $O(N)$     |
| 剑指32-I         | [从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/) | $O(N)$      | $O(N)$     |
| 102 / 剑指32-II  | [二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/) | $O(N)$      | $O(N)$     |
| 103 / 剑指32-III | [从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/) | $O(N)$      | $O(N)$     |
| ==剑指36==       | [二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/) | $O(N)$      | $O(N)$     |
| ==113 / 剑指34== | [路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) | $O(N)$      | $O(N)$     |
| ==剑指33==       | [二叉搜索树的合法后序遍历](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | $O(N^2)$    | $O(N)$     |
| 297 / 剑指37     | [ 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/) | $O(N)$      | $O(N)$     |
| ==剑指26==       | [树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/) | $O(MN)$     | $O(M)$     |
| 124              | [二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/) | $O(N)$      | $O(N)$     |



## 双指针与滑动窗口 (19)

| 题目ID             | 题目名称                                                     | 时间复杂度 | 空间复杂度    |
| ------------------ | ------------------------------------------------------------ | ---------- | ------------- |
| 21                 | [合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/) | $O(M+N)$   | $O(1)$        |
| 剑指21             | [调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/) | $O(N)$     | $O(1)$        |
| 剑指57             | [和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/) | $O(N)$     | $O(1)$        |
| ==151 / 剑指58-I== | [颠倒字符串中的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/) | $O(N)$     | $O(N)$        |
| 剑指58-II          | [左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/) | $O(N)$     | $O(N)$        |
| 160 / 剑指52       | [相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) | $O(M+N)$   | $O(1)$        |
| 876                | [链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) | $O(N)$     | $O(1)$        |
| 剑指22             | [链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/) | $O(N)$     | $O(1)$        |
| 19                 | [删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) | $O(N)$     | $O(1)$        |
| 141                | [环形链表](https://leetcode-cn.com/problems/linked-list-cycle/) | $O(N)$     | $O(1)$        |
| ==142==            | [环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) | $O(N)$     | $O(1)$        |
| 剑指57             | [和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/) | $O(N)$     | $O(1)$        |
| 剑指57-II          | [和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/) | $O(N)$     | $O(1)$        |
| ==15==             | [三数之和](https://leetcode-cn.com/problems/3sum/)           | $O(N^2)$   | $O(1)$        |
| 3 / 剑指48         | [无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/) | $O(N)$     | $O(|\Sigma|)$ |
| 206                | [反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) | $O(N)$     | $O(1)$        |
| 92                 | [反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/) | $O(N)$     | $O(1)$        |
| 25                 | [K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/) | $O(N)$     | $O(1)$        |
| ==143==            | [重排链表](https://leetcode-cn.com/problems/reorder-list/)   | $O(N)$     | $O(1)$        |



## 优先队列 (3)

| 题目ID             | 题目名称                                                     | 时间复杂度    | 空间复杂度 |
| ------------------ | ------------------------------------------------------------ | ------------- | ---------- |
| ==23==             | [合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/) | $O(N*logK)$   | $O(K)$     |
| ==239 / 剑指59-I== | [滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/) | $O(N * logN)$ | $O(N)$     |
| ==295 / 剑指41==   | [数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/) | $O(logN)$     | $O(N)$     |



## 排序 (9)

**[七种排序总结文章](https://leetcode-cn.com/problems/sort-an-array/solution/dong-hua-mo-ni-yi-ge-kuai-su-pai-xu-wo-x-7n7g/)**

| 题目ID     | 题目名称                                                     | 时间复杂度     | 空间复杂度 |
| ---------- | ------------------------------------------------------------ | -------------- | ---------- |
| 912        | 直接插入排序                                                 | $O(N^2)$       | $O(1)$     |
| 912        | 希尔排序                                                     | $O(n^{1.3-2})$ | $O(1)$     |
| 912        | 冒泡排序                                                     | $O(N^2)$       | $O(1)$     |
| 912        | 简单选择排序                                                 | $O(N^2)$       | $O(1)$     |
| 912        | 堆排序                                                       | $O(N*logN)$    | $O(1)$     |
| 912        | [归并排序数组](https://leetcode-cn.com/problems/sort-an-array/) | $O(N*logN)$    | $O(N)$     |
| ==剑指51== | [数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/) | $O(N*logN)$    | $O(N)$     |
| ==912==    | [快速排序数组](https://leetcode-cn.com/problems/sort-an-array/) | $O(N*logN)$    | $O(logN)$  |
| ==215==    | [数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/) | $O(N)$         | $O(logN)$  |



## 动态规划 (23)

| 题目ID                  | 题目名称                                                     | 时间复杂度  | 空间复杂度 |
| ----------------------- | ------------------------------------------------------------ | ----------- | ---------- |
| 剑指10-I                | [斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/) | $O(N)$      | $O(1)$     |
| 70 / 剑指10-II          | [爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)  | $O(N)$      | $O(1)$     |
| ==343== / 剑指14-I & II | [整数拆分](https://leetcode-cn.com/problems/integer-break/)  | $O(N^2)$    | $O(N)$     |
| 剑指47                  | [礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/) | $O(N^2)$    | $O(N^2)$   |
| ==264== / 剑指49        | [丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)  | $O(N)$      | $O(N)$     |
| 53 / 剑指49             | [最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/) | $O(N)$      | $O(1)$     |
| 152                     | [乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) | $O(N)$      | $O(1)$     |
| 121 / 剑指63            | [买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) | $O(N)$      | $O(1)$     |
| ==122==                 | [买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) | $O(N)$      | $O(1)$     |
| ==123==                 | [买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/) | $O(N)$      | $O(1)$     |
| 198                     | [打家劫舍](https://leetcode-cn.com/problems/house-robber/)   | $O(N)$      | $O(N)$     |
| ==42==                  | [接雨水](https://leetcode-cn.com/problems/trapping-rain-water/) | $O(N)$      | $O(N)$     |
| 剑指 46                 | [数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/) | $O(N)$      | $O(N)$     |
| ==5==                   | [最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/) | $O(N^2)$    | $O(N^2)$   |
| ==233 / 剑指43==        | [数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/) | $O(length)$ | $O(1)$     |
| 剑指60                  | [n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/) | $O(N^2)$    | $O(N)$     |
| 剑指19                  | [正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/) | 梭哈        | 梭哈       |
| ==72==                  | [编辑距离](https://leetcode-cn.com/problems/edit-distance/)  | $O(M*N)$    | $O(M*N)$   |
| 1143                    | [最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) | $O(M*N)$    | $O(M*N)$   |
| ==300== **(重点!!!)**   | [LIS: 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) | $O(N*logN)$ | $O(N)$     |
| 64                      | [最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) | $O(N)$      | $O(1)$     |
| 221                     | [最大正方形](https://leetcode-cn.com/problems/maximal-square/) | $O(M*N)$    | $O(M*N)$   |
| 139                     | [单词拆分](https://leetcode-cn.com/problems/word-break/)     | $O(N^2)$    | $O(N)$     |



## 搜索 & 回溯 (5)

| 题目ID | 题目名称                                                     | 时间复杂度 | 空间复杂度 |
| ------ | ------------------------------------------------------------ | ---------- | ---------- |
| 200    | [岛屿数量](https://leetcode-cn.com/problems/number-of-islands/) | $O(M*N)$   | $O(M*N)$   |
| 695    | [岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/) | $O(M*N)$   | $O(M*N)$   |
| 剑指13 | [机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/) |            |            |
| 46     | [全排列](https://leetcode-cn.com/problems/permutations/)     | $O(N*N!)$  | $O(N*N!)$  |
| 54     | [螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)  | $O(M*N)$   | $O(1)$     |



## 二分查找 (8)

| 题目ID            | 题目名称                                                     | 时间复杂度 | 空间复杂度 |
| ----------------- | ------------------------------------------------------------ | ---------- | ---------- |
| 704               | [二分查找](https://leetcode-cn.com/problems/binary-search/)  | $O(logN)$  | $O(1)$     |
| ==34 / 剑指53-I== | [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | $O(logN)$  | $O(1)$     |
| 剑指53-II         | [0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/) | $O(logN)$  | $O(1)$     |
| 240 / 剑指04      | [搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/) | $O(M+N)$   | $O(1)$     |
| ==154 / 剑指11==  | [寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/) | $O(logN)$  | $O(1)$     |
| ==33==            | [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) | $O(logN)$  | $O(1)$     |
| ==4==             | [寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) | $O(M+N)$   | $O(1)$     |
| 69                | [x 的平方根 ](https://leetcode-cn.com/problems/sqrtx/)       | $O(logN)$  | $O(1)$     |



## 栈 (2)

| 题目ID | 题目名称                                                     | 时间复杂度 | 空间复杂度 |
| ------ | ------------------------------------------------------------ | ---------- | ---------- |
| 20     | [有效的括号](https://leetcode-cn.com/problems/valid-parentheses/) | $O(N)$     | $O(N)$     |
| 146    | [LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)      |            |            |
|        |                                                              |            |            |



## 设计/模拟 (1)

| 题目ID  | 题目名称                                                | 时间复杂度 | 空间复杂度    |
| ------- | ------------------------------------------------------- | ---------- | ------------- |
| ==146== | [LRU 缓存](https://leetcode-cn.com/problems/lru-cache/) | $O(1)$     | $O(capacity)$ |
|         |                                                         |            |               |
|         |                                                         |            |               |


#Ques-1

def semordnilap(words):
    # Write your code here
    equal_pairs = []
    for i in range(len(words)):
        s = words[i]
        for j in range(i + 1, len(words)):
            if s == words[j][::-1]:
                       equal_pairs.append([words[i], words[j]])
                       break
        
                       
    return equal_pairs


#Ques2: First Non-Repeating Charector

def firstNonRepeatingCharacter(string):
    # Write your code here
    my_dictionary = {}
    for i in range(len(string)):
        my_dictionary[string[i]] = 0


    for k in range(len(string)):
        my_dictionary[string[k]] = my_dictionary[string[k]] + 1

    for j in range(len(string)):
        if my_dictionary[string[j]] == 1:
            return j
    return -1


#Ques-3
def generateDocument(characters, document):
    # Write your code here.

    for i in range(len(document)):
        if(document.count(document[i]) > characters.count(document[i])):
            return False

    return True

#Ques4:
def commonCharacters(strings):
    ch_count = {}
    for string in strings:
        unique_ch = set(string)
        for ch in unique_ch:
            if ch not in ch_count:
                ch_count[ch] = 0
            ch_count[ch] +=1

    final_ch=[]
    for ch, count in ch_count.items():
        if count == len(strings):
            final_ch.append(ch)

    return final_ch


#Question 5:
def runLengthEncoding(string):
    encoded_str = []
    current_run = 1
    for i in range(1, len(string)):
        current_ch = string[i]
        prev_ch = string[i-1]
        if current_ch != prev_ch or current_run ==9:
            encoded_str.append(str(current_run))
            encoded_str.append(prev_ch)
            current_run = 0
        current_run +=1
    encoded_str.append(str(current_run))
    encoded_str.append(string[len(string) - 1])
    return "".join(encoded_str)

#Ques 6:
def binarySearch(array, target):
    start = 0
    end = len(array) - 1
    middle = (start + end)//2 
    while(start <= end):
        if array[middle] == target:
            return middle
        if array[middle] > target:
            end = middle - 1
        if array[middle] < target:
            start = middle + 1
        middle = (start + end)//2
    return -1


#Ques7:
# This is the class of the input root. Do not edit it.
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def branchSums(root):
    sum_list = []
    sum=0
    sum_list = calculate_list(root,sum,sum_list)
    return sum_list
def calculate_list(root, sum, sum_list):
    if(root is None):
        return sum_list
    sum+=root.value
    if(root.left is None and root.right is None):
        sum_list.append(sum)
        return sum_list
    calculate_list(root.left,sum,sum_list)
    calculate_list(root.right,sum,sum_list)
    return sum_list



#Ques 8:
def nodeDepths(root):
    node_depth = 0
    list_node = []
    calNodeDepth(root, node_depth,list_node)
    return sum(list_node)

def calNodeDepth(root, node_depth,list_node):
    if root is None:
        return 
    list_node.append(node_depth)
    node_depth+=1
    calNodeDepth(root.left, node_depth,list_node) 
    calNodeDepth(root.right,node_depth,list_node)
    return
# This is the class of the input binary tree.
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


#Ques 9:
def classPhotos(redShirtHeights, blueShirtHeights):
    # Write your code here.

    sorted_red = sorted(redShirtHeights, reverse=True)
    sorted_blue = sorted(blueShirtHeights,reverse=True)
    k=0
    while(k < len(sorted_blue) and sorted_red[k] == sorted_blue[k]):
        k+=1
    if k == len(sorted_blue):
        return False
    if sorted_red[k] > sorted_blue[k]:
        for i in range(len(sorted_red)):
            if sorted_red[i] <= sorted_blue[i]:
                return False
    else:
        for j in range(len(sorted_blue)):
            if sorted_blue[j] <= sorted_red[j]:
                return False
    return True



#Ques10:

# This is an input class. Do not edit.
class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def middleNode(linkedList):
    # Write your code here.
    if linkedList.next is None:
        return linkedList
    slow = linkedList
    fast = slow
    if(slow.next is not None and slow.next.next is None):
        return slow.next
    if(slow.next is not None and slow.next.next is not None):
        fast = slow.next.next

    while(fast.next is not None and fast.next.next is not None):
        slow = slow.next
        fast = fast.next.next

    if(fast.next is None):
        return slow.next

    return slow.next.next


#Question 11:
def evaluateExpressionTree(tree):
    # Write your code here.
    if tree.value >= 0:
        return tree.value
    leftValue = evaluateExpressionTree(tree.left)
    rightValue = evaluateExpressionTree(tree.right)

    if tree.value == -1:
        return leftValue + rightValue
    if tree.value == -2:
        return leftValue - rightValue
    if tree.value == -3:
        return int(leftValue/rightValue)
    return leftValue*rightValue



#Ques12:
# This is an input class. Do not edit.
class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def removeDuplicatesFromLinkedList(linkedList):
    # Write your code here.
    if linkedList is None:
        return None
    current_node = linkedList
    while current_node.next is not None:
        if current_node.value == current_node.next.value:
            current_node.next = current_node.next.next
        else:
            current_node = current_node.next
    return linkedList


#Ques13:
ef minimumWaitingTime(queries):
    # Write your code here.
    if len(queries) == 0 or len(queries) == 1:
        return 0
    queries.sort()
    sum = 0
    min = 0
    for i in range(1,len(queries)):
        sum = sum + queries[i-1]
        min = min + sum
    return min


def tournamentWinner(competitions, results):
    # Write your code here.
    if len(results) == 0:
        return ""
    dict = {}

    for i in range(len(results)):
        if competitions[i][0] not in dict:
            dict.update({competitions[i][0]: 0})
        if competitions[i][1] not in dict:
            dict.update({competitions[i][1]: 0})
        if results[i] == 1:
            value = dict[competitions[i][0]]
            dict[competitions[i][0]] = value + 3
        else:
            value = dict[competitions[i][1]]
            dict[competitions[i][1]] = value + 3 
    max_key = max(dict, key=dict.get)
    return max_key

#Ques14:
# Tip: You can use the type(element) function to check whether an item
# is a list or an integer.
def productSum(array):
    # Write your code here.
    return productSum2(array,1)

def productSum2(array, increment):
    if(len(array) == 0):
        return 0
    if(len(array) == 1 and not isinstance(array[0], list)):
        return array[0]
    sum = 0
    for i in range(len(array)):
        if isinstance(array[i], list):
            sum = sum + (increment+1)*productSum2(array[i],increment+1)
        else:
            sum+=array[i]
    return sum
    

#Ques15:

def isPalindrome(string):
    # Write your code here.
    if len(string) == 0:
        return False
    if(len(string)) == 1:
        return True
    i = 0
    j = len(string) - 1

    while(i<j):
        if string[i] != string[j]:
            return False
        i+=1
        j-=1
    return True

#Ques16:
def findThreeLargestNumbers(array):
    # Write your code here.
    first_three = array[:3]
    first_three.sort()
    for i in range(3,len(array)):
        if( array[i] > first_three[2]):
            first_three[0] = first_three[2]
            first_three[2] = array[i]
            first_three.sort()
        elif(array[i] > first_three[1]):
            first_three[0]=first_three[1]
            first_three[1] = array[i]
        elif(array[i] > first_three[0]):
            first_three[0] = array[i]
    return first_three


#Ques17:
def isValidSubsequence(array, sequence):
    # Write your code here.
    j=0
    for i in range(len(sequence)):
        found = False
        for k in range(j,len(array)):
            j+=1
            if(sequence[i] == array[k]):
                found = True
                break
        if not found:
            return False
    return True


#Ques18:
def getNthFib(n):
    # Write your code here.
    if(n == 1):
        return 0
    if(n == 2):
        return 1
    return getNthFib(n-1) + getNthFib(n-2)

#Ques 19:
def caesarCipherEncryptor(string, key):
    # Write your code here.
    ch = "abcdefghijklmnopqrstuvwxyz"
    string_list = list(string)
    for i in range(len(string_list)):
        k = (ch.find(string[i]) + key)%26
        string_list[i]=ch[k]
    return ''.join(string_list)

#Ques20:
def sortedSquaredArray(array):
    # Write your code here.
    for i in range(len(array)):
        array[i] = array[i]*array[i]
    array.sort()
    return array

#Ques21:
def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):
    # Write your code here.
    min = 0
    maximum = 0
    redShirtSpeeds.sort()
    blueShirtSpeeds.sort()
    length = len(blueShirtSpeeds)
    for i in range(len(blueShirtSpeeds)):
        min = min + max(redShirtSpeeds[i],blueShirtSpeeds[i])
        maximum = maximum + max(redShirtSpeeds[i],blueShirtSpeeds[length-1-i])

    if fastest:
        return maximum
    return min


#Ques22:
def twoNumberSum(array, targetSum):
    # Write your code here.
    dict = {}
    for i in range(len(array)):
        dict[array[i]] = i

    for j in range(len(array)):
        rem = targetSum - array[j]
        if rem != array[j]:
            if rem in dict:
                return [array[j], rem]
    return []

#Ques23:
def nonConstructibleChange(coins):
    # Write your code here.
    if(len(coins)==0):
        return 1
    coins.sort()
    create = 0
    for i in range(len(coins)):
        if coins[i] > create + 1:
            return create + 1
        if coins[i] <= create + 1:
            create = create + coins[i]

    return create + 1

#Ques24:
def bubbleSort(array):
    # Write your code here.
    is_sorted=False
    while not is_sorted:
        is_sorted= True
        for i in range(len(array)-1):
            if(array[i] > array[i+1]):
                temp = array[i]
                array[i]= array[i+1]
                array[i+1]=temp
                is_sorted = False
    return array

#Ques25:
# Do not edit the class below except
# for the depthFirstSearch method.
# Feel free to add new properties
# and methods to the class.
class Node:
    def __init__(self, name):
        self.children = []
        self.name = name

    def addChild(self, name):
        self.children.append(Node(name))
        return self

    def depthFirstSearch(self, array):
        # Write your code here.
        array.append(self.name)
        for i in self.children:
            i.depthFirstSearch(array)
        return array

#Ques:26
def findClosestValueInBst(tree, target):
    closest = float("inf")
    current=tree
    while current is not None:
        if abs(target - closest) > abs(target - current.value):
            closest = current.value
        if(current.value < target):
            current = current.right
        elif(current.value > target):
            current = current.left
        else:
            break
    return closest


# This is the class of the input tree. Do not edit.
class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

#Quest27:
def firstDuplicateValue(array):
    # Write your code here.

    for value in array:
        absValue = abs(value)
        if(array[absValue-1] < 0):
            return absValue
        array[absValue - 1]*=-1
    return -1
#Ques28:
def validIPAddresses(string):
    # Write your code here.
    validIp = []

    for i in range(1,min(len(string),4)):
        currentIp = ['','','','']
        currentIp[0]=string[:i]
        if not isvalid(currentIp[0]):
            continue
        for j in range(i+1, i + min(len(string)-i,4)):
            currentIp[1] = string[i:j]
            if not isvalid(currentIp[1]):
                continue
            for k in range(j+1, j + min(len(string)-j,4)):
                currentIp[2]=string[j:k]
                currentIp[3]=string[k:]

                if isvalid(currentIp[2]) and isvalid(currentIp[3]):
                    validIp.append('.'.join(currentIp))
    
    return validIp
def isvalid(strr):
    value = int(strr)
    if value > 255:
        return False
    return len(strr) == len(str(value))

#Ques29:
def numberOfWaysToMakeChange(n, denoms):
    # Write your code here.
    ways = [0 for amount in range(n+1)]
    ways[0] = 1
    for denom in denoms:
        for amount in range(1,n+1):
            if(denom <= amount):
                ways[amount]+=ways[amount-denom]

    return ways[n]

#Ques30:
def smallestDifference(arrayOne, arrayTwo):
    arrayOne.sort()
    arrayTwo.sort()
    i = 0
    j = 0
    smallest = float("inf")
    current = float("inf")
    pair=[]
    while i < len(arrayOne) and j < len(arrayTwo):
        one = arrayOne[i]
        two = arrayTwo[j]
        if one < two:
            i+=1
            current = two - one
        elif two < one:
            j+=1
            current = one-two
        else:
            return [one,two]
        if smallest > current:
            smallest = current
            pair = [one,two]
    return pair


#Ques31:
For non contiguous sub array problem:
def zeroSumSubarray(nums):
    # Write your code here.
    if len(nums)==0:
        return False
    elif len(nums)==1 and nums[0]==0:
        return True
    dict={}
    dpsum =[0 for num in range(len(nums))]
    dpsum[0]=nums[0]
    dict.update({nums[0]:0})
    for i in range(1,len(nums)):
        dpsum[i] = dpsum[i-1] + nums[i]
        if nums[i] not in dict:
            dict.update({nums[i]:i})
        if dpsum[i] in dict:
            return True
        if nums[i] == 0:
            return True
    return False

#Ques31b:
def zeroSumSubarray(nums):
    sums = set([0])
    currentSum = 0
    for num in nums:
        currentSum+=num
        if currentSum in sums:
            return True
        sums.add(currentSum)
    return False

#Ques32:
def isMonotonic(array):
    # Write your code here.
    i=0
    j=0
    dec=False
    n = len(array)
    if (n==0):
        return True
    while(j < n and array[i]==array[j]):
        j+=1
    if j < n and array[i] > array[j]:
        dec = True
    elif j ==n:
        return True
    if dec:
        for i in range(j,n-1):
            if array[i] < array[i+1]:
                return False
    else:
        for i in range(j,n-1):
            if array[i] > array[i+1]:
                return False
        
    return True

#Ques33:
def getPermutations(array):
    # Write your code here.
    permutations=[]
    getPerm(array,[],permutations)
    return permutations
def getPerm(array,current,permutations):
    if len(array) == 0 and len(current) != 0:
        permutations.append(current)
    else:
        for i in range(len(array)):
            newArray = array[:i] + array[i+1:]
            newcurrent = current + [array[i]]
            getPerm(newArray,newcurrent,permutations)

#Ques34:
# This is an input class. Do not edit.
class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def sumOfLinkedLists(linkedListOne, linkedListTwo):
    # Write your code here.
    s1 = ""
    s2=""
    while(linkedListOne is not None):
        s1= s1 + str(linkedListOne.value)
        linkedListOne = linkedListOne.next
    while(linkedListTwo is not None):
        s2= s2 + str(linkedListTwo.value)
        linkedListTwo = linkedListTwo.next
    sum = int(s1[::-1]) + int(s2[::-1])
    s3 = str(sum)
    s3= s3[::-1]
    head = LinkedList(int(s3[0]))
    current = head
    for i in range(1,len(s3)):
        node = LinkedList(int(s3[i]))
        current.next = node
        current = node
    
    return head

#Ques35:
def oneEdit(stringOne, stringTwo):
    # Write your code here.
    n1 = len(stringOne)
    n2 = len(stringTwo)
    if(abs(n1-n2) > 1):
        return False
    if(n1==1 and n2==1):
        return True
    count = 0
    if n1==n2:
        for i in range(len(stringOne)):
            if(stringOne[i] != stringTwo[i]):
                count+=1
            if(count>1):
                return False
    elif n1>n2:
        return isSubSequence(stringOne,stringTwo,n1)
    else: 
        return isSubSequence(stringTwo,stringOne,n2)
        
    return True

def isSubSequence(big,small,n1):
    count = 0
    j = 0
    i=0
    while(i < n1-1 and j < n1):
        if(small[i] != big[j]):
            count+=1
            j+=1
            continue
        i+=1
        j+=1
    return not count > 1

#Ques36:
def balancedBrackets(string):
    # Write your code here.
    dict={'}':'{',']':'[',')':'('}
    dict2={'{':'}','[':']','(':')'}
    n = len(string)
    closeList = []
    for i in range(n):
        if(string[n-i-1] not in dict and string[n-i-1] not in dict2):
            continue
        if string[n-1-i] in dict:
            closeList.append(string[n-1-i])
        else:
            if(len(closeList) == 0):
                return False
            if(string[n-1-i] in dict2 and dict2[string[n-1-i]] != closeList.pop()):
               return False
    return len(closeList) == 0

#Ques37:
def missingNumbers(nums):
    # Write your code here.
    nums.sort()
    ans=[]
    j = 1
    i = 0
    while(i < len(nums)):
        if(nums[i] != j):
            ans.append(j)
            j+=1
            continue
        i+=1
        j+=1
    if len(ans) == 0:
        ans.append(j)
        ans.append(j+1)
    if len(ans) == 1:
        ans.append(j)
    return ans

#Ques38:
def staircaseTraversal(height, maxSteps):
    # Write your code here.
    dp = [0]*(height+1)
    dp[0] = 1
    dp[1] = 1
    for k in range(2,maxSteps+1):
        dp[k] = sumdp(k-1,dp)
    for i in range(maxSteps+1,height+1):
        sum=0
        for j in range (maxSteps):
            sum+=dp[i-j-1]
        dp[i]=sum
        sum=0
    
    return dp[height]
def sumdp(n,dp):
    sum = 0
    for i in range(n+1):
        sum+=dp[i]
    return sum

#Ques39:
# This is an input class. Do not edit.
class BST:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def findKthLargestValueInBst(tree, k):
    # Write your code here.
    result = []
    findK(tree,result)
    return result[k-1]

def findK(tree,result):
    if tree is None:
        return
    findK(tree.right,result)
    result.append(tree.value)
    findK(tree.left,result)

#Ques40:
def reverseWordsInString(string):
   
    string = list(string)
    n = len(string)
    string = reverseStr(string)
    i = 0
    while i < n:
        if(string[i]==' '):
            i+=1
            continue
        j = i
        while j < n and string[j] != ' ':
            j=j+1
        k = j
        while(i < j-1):
            temp = string[i]
            string[i] = string[j-1]
            string[j-1] = temp
            i+=1
            j-=1
        i=k
    return ''.join(string)
def reverseStr(str):
    i=0
    j=len(str)-1
    while(i<j):
        temp = str[i]
        str[i]=str[j]
        str[j]=temp
        i+=1
        j-=1
    return str
       

#Ques41:
def taskAssignment(k, tasks):
    # Write your code here.
    dict ={}
    for i in range(len(tasks)):
        if tasks[i]  in dict:
            dict[tasks[i]].append(i)
        else:
            dict[tasks[i]] = [i]
    tasks.sort()
    sol = []
    i = 0
    j = len(tasks)-1
    
    while(i < j):
        indexIList = dict[tasks[i]]
        indexI = indexIList.pop()
        indexJList = dict[tasks[j]]
        indexJ = indexJList.pop()
        sol.append([indexI,indexJ])
        i+=1
        j-=1
    return sol

#Ques42:
def nextGreaterElement(array):
    # Write your code here.
    n = len(array)
    result = [-1]*n
    i =0
    while(i < n):
        s = findNext(array,i,n)
        result[i]=s
        i+=1
    return result
def findNext(array,index,n):
    j = (index+1)%n
    while(j != index):
        if(array[j] > array[index]):
            return array[j]
        j=(j+1)%n
    return -1

#Ques43:
def bestSeat(seats):
    # Write your code here.
    sum=0
    index=-1
    i=0
    n=len(seats)
    while i < n:
        if(seats[i]==1):
            i+=1
            continue
        j=i
        while(i < n and seats[i]==0):
            i+=1
        if(i-j > sum):
            sum=i-j
            index=int((i+j-1)/2)
    return index

#Ques44:
def bestDigits(number, numDigits):
    # Write your code here.
    stack = []
    for digit in number:
        while numDigits > 0 and len(stack) > 0 and digit > stack[len(stack)-1]:
            numDigits-=1
            stack.pop()
        stack.append(digit)

    while numDigits >0:
        numDigits-=1
        stack.pop()
    return "".join(stack)



#Ques45:
def threeNumberSort(array, order):
    # Write your code here.
    n = len(array)
    m = len(order)
    k=0
    i = 0
    for j in range(m):
        for i in range(k,n):
            if(array[i] == order[j]):
                temp = array[k]
                array[k]=array[i]
                array[i]=temp
                k+=1
    return array







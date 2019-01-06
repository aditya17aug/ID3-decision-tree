from node import Node
import math
'''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
'''
def ID3(examples, default):
    node = None
    if examples is None:
        node = Node()
        node.label = default
    else:
        node = id3_impl(examples, default)
    return node

'''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
'''   
def prune(node, examples): 
    topAcc = test_impl(node, examples, True)
    updatedAcc = 0.0
    while True:
        allNode = search(node)
        for localnode in allNode:
            prevAccuracy = topAcc
            flag = False
            for child in localnode.children:
                if not localnode.children[child].isLeaf:
                    flag = False
                    break
                else:
                    flag = True
            if flag:                
                localCountDict = {}
                for child in localnode.children:
                    if localnode.children[child].label in localCountDict:
                        localCountDict[localnode.children[child].label] = localCountDict[localnode.children[child].label] + localnode.children[child].numberOfTimeTraversed
                    else:
                        localCountDict[localnode.children[child].label] = localnode.children[child].numberOfTimeTraversed
                famousLabel = None
                key_max = max(localCountDict.keys(), key=(lambda k: localCountDict[k]))
                key_min = min(localCountDict.keys(), key=(lambda k: localCountDict[k]))
                if key_min != key_max:
                    famousLabel = key_max
                if famousLabel is not None:
                    backupLabel = localnode.label
                    backupChild = localnode.children
                    localnode.label = famousLabel
                    localnode.children = {}
                    localnode.setLeafNode(True)
                    newAcc = test_impl(node, examples, False)
                    if prevAccuracy > newAcc :
                        localnode.label = backupLabel
                        localnode.children = backupChild
                        localnode.setLeafNode(False)
                    else:
                        prevAccuracy = newAcc
                        updatedAcc = prevAccuracy
        if topAcc >= updatedAcc:
            break
        else:
            topAcc = updatedAcc

'''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
'''
def test(node, examples):
    return test_impl(node, examples)

'''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
'''
def evaluate(node, example):
      return evaluate_impl(node, example)

def test_impl(node, examples, isTraining=False):
    lengthOfTestData = len(examples)
    totalCorrect = 0
    for example in examples:
        if evaluate_impl(node, example, isTraining) ==  example["Class"]:
            totalCorrect = totalCorrect + 1;
    acc = totalCorrect/lengthOfTestData
    return acc

def evaluate_impl(node, example, isTraining=False):
    nodeName = node.label
    if node.isLeaf:
        if isTraining:
            node.increment_numberOfTimeTraversed()
        return nodeName
    else:
        value = example[nodeName]
        if value is not '':
            if value not in node.children:
                return node.default
            else:
                if isTraining:
                    node.increment_numberOfTimeTraversed()
                return evaluate_impl(node.children[value], example, isTraining)
        else:
            return node.default  
        
def print_root(node, number_of_tabs):
    print(number_of_tabs,node.label,"(count = ",node.numberOfTimeTraversed,")")
    def print_tree(node, number_of_tabs):
        for eachChild in node.children:
            print(number_of_tabs,"|")
            print(number_of_tabs,"--",eachChild,"--")
            if len(node.children[eachChild].children)>0:
                print(number_of_tabs,"       ",node.children[eachChild].label,"(count = ",node.children[eachChild].numberOfTimeTraversed,")")
                print_tree(node.children[eachChild], number_of_tabs + "\t")
            else:
                print(number_of_tabs,"        ",node.children[eachChild].label,"(count = ",node.children[eachChild].numberOfTimeTraversed,")")
    print_tree(node, number_of_tabs)
    

def id3_impl(data, default):
    leafNodesList = []
    
    allAttribute = data_seprator(data)

    leafNodes = allAttribute.get("Class")
    for eachLeaf in leafNodes:
        node = Node() 
        node.label = eachLeaf
        node.setLeafNode(True)
        leafNodesList.append(node)
  
    attributeInfoGain = {}
    for key in allAttribute:
        if key != "Class":
            attributeInfoGain[key] = info_gain_calculator(allAttribute.get(key), allAttribute.get("Class"))
    
    nodeName = ""
    highestInfoGain = 0.0
    for key in attributeInfoGain :
        infogain = attributeInfoGain.get(key) 
        if infogain > highestInfoGain or (infogain == 0.0 and highestInfoGain == 0.0):
            highestInfoGain = infogain
            nodeName = key
    node = Node() 
    node.label = nodeName
    node.setDefault(default)
    prepare_childs(node, allAttribute, data)
    return node
    

def prepare_childs(node, allAttribute, data):
    childLabel = "";
    nodeChilds = node.children
    for key in filter(lambda w: w in node.label, allAttribute):
        childLabel = key
    if childLabel is '':
        return
    values = allAttribute[childLabel]
    for eachValue in values:
        child = Node()
        child.setDefault(node.default)
        if len(values[eachValue]) == 1:
            for eachLeaf in values[eachValue]:
                child.label = eachLeaf
                child.setLeafNode(True)
            nodeChilds[eachValue] = child
        else:
            listOfAttribute = {}
            listOfAttribute[childLabel] = eachValue
            newData = data_filteration(data, listOfAttribute)
            if len(newData) == 0:
                continue
            else: 
                nodeChilds[eachValue] = id3_impl(newData, node.default)
    
def data_filteration(data, listOfAttribute):
    updatedData = []
    for eachData in data:
        for eachRestriction in listOfAttribute:
            if eachData[eachRestriction] == listOfAttribute[eachRestriction]:
                updatedData.append(eachData)
    if len(listOfAttribute) == 0:
        return []
    return updatedData
    
def data_seprator(data):
    entropyOfAllAttribute = {}
    for eachData in data:
        className = eachData.get("Class")  
        for key in eachData:
            if key in entropyOfAllAttribute.keys():
                attributes = entropyOfAllAttribute.get(key)
            else:
                attributes = {}
                entropyOfAllAttribute[key] = attributes
            if key != "Class":
                if eachData.get(key) not in attributes.keys(): 
                    attributes[eachData.get(key)] = {}
                value  = attributes.get(eachData.get(key))
                if className in value.keys():
                    count  = value.get(className)
                    value[className] = count+1
                else:
                    value[className] = 1                    
            else:
                if eachData.get(key) in attributes.keys():
                    value  = attributes.get(eachData.get(key))
                    attributes[eachData.get(key)] = value+1
                else:
                    attributes[eachData.get(key)] = 1
    return entropyOfAllAttribute
                
def entropy_calculator(attributes):
    entropy = 0.0
    totalClassesSum = sum(attributes.values())
    for eachClassAttribute in attributes:
        probabaility = attributes.get(eachClassAttribute)/totalClassesSum
        entropy -= probabaility * math.log2(probabaility)
    return entropy
    
def info_gain_calculator(attributes, classAttribute):
    numberOfClassAttribute = len(classAttribute)
    infoGainData = {}
    for key in attributes:
        if len(attributes.get(key)) < numberOfClassAttribute:
            ig = 0
        else:
            ig = entropy_calculator(attributes.get(key))
        infoGainData[key] = ig
    attr_entropy = 0
    totalClassesSum = sum(classAttribute.values())
    totalAtributeSum = 0
    for key in attributes:
        dic = attributes.get(key)
        totalAtributeSum = sum(dic.values())
        totalAtributeSum = (totalAtributeSum/totalClassesSum)*infoGainData.get(key)
        attr_entropy = attr_entropy + totalAtributeSum  
    gain = entropy_calculator(classAttribute) - attr_entropy
    return gain

def get_all_leaf_nodes(node):
    leafs = []
    def get_all_leaf(node):
        if node is not None:
            if not node.children:
                leafs.append(node.label)
            for n in node.children:
                get_all_leaf(node.children[n])
    get_all_leaf(node)
    return leafs

        
def search(tree):
    queue = [tree]
    visited = []
    while queue:
        vertex = queue.pop(0)
        if vertex.isLeaf:
            continue
        visited.append(vertex)
        for w in vertex.children:
            queue.append(vertex.children[w]) 
    return visited
class Node:
  def __init__(self):
    self.label = None
    self.children = {}
    self.isLeaf = False
    self.default = ""
    self.numberOfTimeTraversed = 0
    
  def setLeafNode(self, isLeaf):
    self.isLeaf = isLeaf

  def setDefault(self, default):
      self.default = default
      
  def increment_numberOfTimeTraversed(self):
      self.numberOfTimeTraversed = self.numberOfTimeTraversed + 1
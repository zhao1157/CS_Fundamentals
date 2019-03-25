#======== 26 ========

def find_all_paths(graph, start, end, path=[]):
	print ('1', path)
	path = path + [start] #path.append(start) is a in-place modification, whiel path += [start] is not.
	print ('2', path)
	if start == end:
		return [path]
        #if not graph.has_key(start):
	if start not in graph.keys():
		return []
	paths = []
	for node in graph[start]:
		if node not in path:
			newpaths = find_all_paths(graph, node, end, path)
			for newpath in newpaths:
				paths.append(newpath)
	return paths

graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
             
print (find_all_paths(graph, 'A', 'D'))

def Paths(graph, start, end, path=[]):
	print ('1', path)
	#path.append(start)
	path = path + [start]
	print('2', path)
	if start == end:
		return [path]
	if start not in graph:
		return []
	
	all_path = []	
	for node in graph[start]:
		if node not in path:
			new_path = Paths(graph, node, end, path)
			#print ('2', node, path)
			for i in new_path:
				all_path.append(i)
				
	return all_path		
	
path = []
#all_path = Paths(graph, 'A', 'D', path)
#print (all_path)
print ('----')
def find_all_path(graph, start, end, path = [], coll=[]):
	path.append(start)
	if start == end:
		return True
	if start not in graph.keys():
		return False
	
	for node in graph[start]:
		if node not in path:
			found = find_all_path(graph, node, end, path, coll)
			if found:
				coll.append(path.copy()) #super important
			path.pop()


	return False
	
paths=[]
print (find_all_path(graph, 'A','D', [], paths))
print (paths)




#========= 25 =========
#This is to practice dfs
def vist_graph(graph, start, visited=None):
	if visited == None:
		visited = set()
	print (start, end = ' ')
	visited.add(start)
	for i in graph[start]-visited:
		try:
			vist_graph(graph, i, visited)
		except:
			print ('error')
		
graph = {'a': set(['b', 'c', 'e']),
		 'b': set(['a', 'c', 'd']),
		 'c': set(['e', 'd'])}

#vist_graph(graph, 'a')


def visit_bfs(graph, start, visited = None):
	if visited is None:
		visited = set()
		visited.add(start)
	queue = []
	print (visited)
	for i in graph[start]-visited:
		visited.add(i)
		queue.insert(0, i)
	for i in queue:
		visit_bfs(graph, i, visited)

graph = {'a': set(['b']),
		 'b': set(['a', 'c', 'd']),
		 'c': set(['e', 'd']),
		 'd': set(['b', 'a']),
		 'e': set(['c', 'a'])}

visited=None
visit_bfs(graph, 'a', visited)
print (visited)




#======= 24 ======
#This is to practice Dijkstra's algorithm
class Heap:
	def __init__(self, array):
		self.array = array
		self.max_parent = (len(self.array)-2)//2
		self.max_index = len(self.array) -1
	
	def get_array(self):
		return self.array
		
	def heapify(self, ind_parent): #up - bottom
		if ind_parent <= self.max_parent: #at least one child(left)
			ind_small = ind_parent
			if self.array[ind_small][1] > self.array[2*ind_parent+1][1]:
				ind_small = 2*ind_parent+1
			if 2*ind_parent+2 <= self.max_index and self.array[ind_small][1] > self.array[2*ind_parent+2][1]:
				ind_small = 2*ind_parent+2
			if ind_small != ind_parent:
				self.array[ind_parent], self.array[ind_small] = self.array[ind_small], self.array[ind_parent]
				self.heapify(ind_small) #don't forget it.
				
	def build_heap(self):
		for parent in range(self.max_parent, -1, -1):
			self.heapify(parent)
	
	def heapify_up(self, ind, new_value):
		self.array[ind] = new_value
		while ind > 0:
			parent = (ind-1)//2
			if self.array[ind][1] < self.array[parent][1]:
				self.array[ind], self.array[parent] = self.array[parent], self.array[ind]
			ind = parent
				
	def heapop(self): #return and remove the smallest value
		self.array[0], self.array[-1] = self.array[-1], self.array[0]
		min_value = self.array.pop()
		
		self.max_parent = (len(self.array)-2)//2
		self.max_index = len(self.array) -1
		
		self.heapify(0)
		return min_value

class Vertex:
	def __init__(self, key):
		self.id = key
		self.connections = {} #nbr:weight
		self.distance = 10**10 #set a large number that is larger than any possible value in the problem
		self.pred = None
		
	def add_neighbor(self, nbr, weight = 0):	
		self.connections[nbr] = weight 
	
	def get_connections(self):
		return self.connections.keys()

	def get_distance(self):
		return self.distance
	
	def set_distance(self, dist):
		self.distance = dist
	
	def get_id (self):
		return self.id
	def get_pred(self):
		return self.pred
	def set_pred(self, pred):
		self.pred = pred 
	
class DijkstraGraph:
	def __init__(self):
		self.vert_list = {}
		
	def add_vertex(self, key):
		self.vert_list[key] = Vertex(key)
	
	def add_edges(self, start, end, weight):
		if start not in self.vert_list:
			self.add_vertex(start)
		if end not in self.vert_list:
			self.add_vertex(end)
		self.vert_list[start].add_neighbor(self.vert_list[end], weight)
		self.vert_list[end].add_neighbor(self.vert_list[start], weight)
	
	def dijkstra_sort(self, start):
		self.vert_list[start].set_distance(0)
		queue_dist = Heap([(self.vert_list[key], self.vert_list[key].get_distance()) for key in vert_list])
		queue_dist.build_heap()
		while len(queue_dist) > 0:
			current_vert = queue_dist.heapop()[0] #extract the vertex
			for nbr in current_vert.get_connections():
				if current_vert.get_distance() + current_vert.connections[nbr] < nbr.get_distance():
					nbr.set_distance = current_vert.get_distance() + current_vert.connections[nbr]
					nbr.set_pred(current_vert)
					for i in range(len(queue_dist.get_array())):
						if queue_dist.get_array()[i][0] is nbr:
							break	
					queue_dist.heapify_up(i, nbr.get_distance())
				


#========= 23 ========
#This is to practice Knight's Tour graph
class Vertex:
	def __init__(self, key):
		self.id = key
		self.connections = {}
		
		self.color = 'white'
		self.distance = 0
		self.pred = None
		self.discover_time = 0
		self.finish_time = 0
	
	def get_id(self):
		return self.id
		
	def set_color(self, color):
		self.color = color
	def get_color(self):
		return self.color 
	
	def set_distance(self, dist):
		self.distance = dist
	def get_distance(self):
		return self.distance
	
	def set_pred(self, pred):
		self.pred = pred 
	def get_pred(self):
		return self.pred 
	
	def set_finish_time(self, time):
		self.finish_time = time
	def get_finish_time(self):
		return self.finish_time
	
	def set_discover_time(self, time):
		self.discover_time = time
	def get_discover_time(self):
		return self.discover_time	
	
	def add_connection(self, vertex, weight=0):
		self.connections[vertex] = weight
	def get_connections(self):
		return list(self.connections.keys())
	
	
	
class KnightsTour:
	def __init__(self, boardsize):
		self.vert_list = {} #key:Vertex(key)
		self.time = 0 #keep track of the steps
		self.boardsize = boardsize
		self.collection = []
		self.found = False
		
	def add_vertex(self, key):
		self.vert_list[key] = Vertex(key)
	def add_edges(self, start, end, weight = 0):
		if start not in self.vert_list:
			self.add_vertex(start)
		if end not in self.vert_list:
			self.add_vertex(end)
		self.vert_list[start].add_connection(self.vert_list[end])
	
	def build_graph(self):
		for row in range(self.boardsize):
			for column in range(self.boardsize):
				center = (row,column)
				nbrs = self.get_nbrs(center)
				for nbr in nbrs:
					self.add_edges(center, nbr)

	def get_nbrs(self, center):
		nbrs = []
		offsets = [(-1, 2), (1, 2), (-2, 1), (2, 1), (-2, -1), (2, -1), (-1, -2), (1, -2)]
		for (x, y) in offsets:
			nbr = (x+center[0], y+center[1])
			if 0<=nbr[0]<self.boardsize and 0<=nbr[1]<self.boardsize:
				nbrs.append(nbr)
		return nbrs
	
	def tour(self, start): #pick a position and start touring
		self.reset_vertex() #reset everything
		for vertex in [self.vert_list[start]]:
			self.dfs(vertex, self.time)
				
	def dfs(self, start_vertex, start_time):
		if start_vertex.get_color == 'white':
			start_vertex.set_color = 'gray'
			start_vertex.set_discover_time(start_time)
			self.time += 1
		for child in start_vertex.get_connections():
			if child.get_color == 'white':
				child.set_pred(start_vertex)
				self.dfs(child, self.time)
		start_vertex.set_finish_time = self.time
		self.time += 1
		start_vertex.set_color('black')
	
	
	
	def reset_vertex(self):
		self.time = 0
		self.collection = []
		for key in self.vert_list:
			self.vert_list[key].set_color('white')
			self.vert_list[key].set_distance(0)
			self.vert_list[key].set_pred(None)
			self.vert_list[key].set_discover_time(0)
			self.vert_list[key].set_finish_time(0)
	
	def knight_tour_action(self, start):
		self.reset_vertex()
		print (self.vert_list[start].get_id())
		for vertex in [self.vert_list[start]]: #just allow only one element!!!
			if vertex.get_color() == 'white':
				self.create_path(vertex) #condition is color of white
				for ver in self.collection:
					print (ver.get_id(), end = ' ')
	def create_path(self, vertex):
		vertex.set_color('gray')
		self.collection.append(vertex)
		
		#for i in self.collection:
		#	print (i.get_id(), end = ', ')
		#print ()
		print (len(self.collection))
		
		if len(self.collection) == self.boardsize*self.boardsize: 
			print (True)
			return True
		else:
			if len(vertex.get_connections()) == 0: #deal with the end leaf
				print ('end leaf')
				return False
			i = 0
			for child in vertex.get_connections(): #deal with the non-end leaf
				i += 1
				if child.get_color() == 'white':
					child.set_pred(vertex)
					self.found = self.create_path(child)
					if self.found == True:
						print ('succeeded') #print many times.
						return True
					else:
						child_reset = self.collection.pop()
						child_reset.set_color ('white')
						child_reset.set_pred(None)
						if i == len(vertex.get_connections())-1: 
							return False #only return false when all the children are tested out.
	
	def print_vert(self, key):
		print (self.vert_list[key].get_connections())	


			
my_tour = KnightsTour(4)
my_tour.build_graph()
print ('--')
for i in my_tour.vert_list[(3, 3)].get_connections():
	print (i.get_id(), end=', ')
print ()
print ('--')
my_tour.knight_tour_action((0, 0))




#======= 22 ========
#This is to solve the word ladder problem
class Vertex:
	def __init__(self, key):
		self.id = key 
		self.connections = {} #key: weight
		self.color = 'white'
		self.distance = 0
		self.pred = None
		
	def add_connection(self, key, weight = 0):
		self.connections[key] = weight
	
	def get_connections(self):
		return self.connections.keys()
		
	def get_id(self):
		return self.id
		
	def get_color(self):
		return self.color
		
	def set_color(self, color):
		self.color = color
	
	def get_distance(self):
		return	self.distance
	
	def set_distance(self, distance):
		self.distance = distance
	
	def get_pred(self):
		return self.pred	
	
	def set_pred(self, pred):
		self.pred = pred
			
		
class WordLadder:
	def __init__(self):
		self.vertices = {}
	
	def add_vertex(self, key):
		self.vertices[key] = Vertex(key)
		
	def add_edges(self, start, end, weight=0):
		if start not in self.vertices:
			self.add_vertex(start)
		if end not in self.vertices:
			self.add_vertex(end)
		self.vertices[start].add_connection(self.vertices[end], weight)
	
	def build_graph(self, word_file):
		word_keys = {}
		#categorize the words
		for line in open(word_file):
			word = line.rstrip()
			for i in range(len(word)):
				key = word[:i]+'_'+word[i+1:]
				if key in word_keys:
					word_keys[key].append(word)
				else:
					word_keys[key] = word 
		
		#construct graph
		for key in word_keys:
			for word_1 in word_keys[key]:
				for word_2 in word_keys[key]:
					if word_1 == word_2:
						self.add_edges(word_1, word_2)
	
	def reset_vertx(self):
		for key in self.vertices:
			self.vertices[key].set_color('white')
			self.vertices[key].set_distance(0)
			self.vertices[key].set_pred(None)
	
	#breadth first search
	def bfs(self, start, end): #start and end are the keys, not the objects
		#reset the color, distance and predecessor
		self.reset_vertx()
	
		#start searching
		queue_inspection = []
		queue_inspection.insert(0, self.vertices[start])
		
		while(len(queue_inspection) > 0):
			pred = queue_inspection.pop()
			for nbr in pred.get_connections():
				if nbr.get_color() == 'white':
					nbr.set_color('gray')
					nbr.set_distance (1+pred.get_distance())
					nbr.set_pred(pred)
					queue_inspection.insert(0, nbr)
					if nbr.get_id() == end:
						print ('Succeeded: path length is '+ str(nbr.get_distance()))
						return (nbr)
			pred.set_color('black')
		print ('Failed')
		return (None)
	
	def get_path(self, vertex_end):
		vertex = vertex_end.copy()
		while vertex:
			print (vertex.get_id(), end = '->')
			vertex = vertex.get_pred()




#======= 21 ========
#This is to practice building vertex and graph
class Vertex:
	def __init__(self, key):
		self.id = key
		self.connectedTo = {} #key: value(weight)
	
	def add_neighbor(self, nbr, weight = 0):	#unweighted and weighted
											#assume the neighbors are Vertex objects.
		self.connectedTo[nbr] = weight
	
	def get_id(self):
		return self.id 
	
	def get_connections(self):
		return list(self.connectedTo.keys())
	
	def get_weights(self, nbr):
		return self.connectedTo[nbr]

	def __str__(self):
		return 'Vertex '+str(self.id) +' has the following neighbors: ' + str([(vert.id, self.connectedTo[vert]) for vert in self.connectedTo])
		
class Graph:
	def __init__(self):
		self.ver_list = {}
		self.num_ver = 0

	def add_vertx (self, key):
		if key not in self.ver_list:
			self.ver_list[key] = Vertex(key)
			self.num_ver += 1
	
	def add_edges(self, start, end, weight = 0):
		if start == end:
			print ('the edge can only be between different vertices')
			return False
			
		if start not in	self.ver_list:
			self.add_vertx(start)
		if end not in self.ver_list:
			self.add_vertx(end)
		self.ver_list[start].add_neighbor(self.ver_list[end], weight)
		
	def get_vertices(self):
		return list(self.ver_list.keys()) #force the output is a list
	
	def print_vert_connections(self, key):
		print (self.ver_list[key])
	
	def __contains__(self, vert): #defines our own "x in Graph" condition
		return vert in self.ver_list		
		
my_graph = Graph()
for i in range(6):
	my_graph.add_vertx(i)

print ('my graph has the following vertices:', str(my_graph.get_vertices()))

my_graph.add_edges(0, 1, 0)
my_graph.add_edges(2, 1, 1)
my_graph.add_edges(3, 2, 1)
my_graph.add_edges(4, 3, 3)
my_graph.add_edges(4, 1, 2)
my_graph.add_edges(5, 2, 5)
my_graph.add_edges(5, 4, 8)

for vertex in my_graph.get_vertices():
	my_graph.print_vert_connections(vertex)

print ('2' in my_graph, 2 in my_graph)



#======= 20 =======
#This is to practice quick sort
def quick_sort(unsorted_array):
	if len(unsorted_array) <= 1:
		return unsorted_array
	elif len(unsorted_array) == 2:
		if unsorted_array[0] > unsorted_array[1]:
			unsorted_array[0], unsorted_array[1] = unsorted_array[1], unsorted_array[0]
		return unsorted_array
	
	pivot_value = unsorted_array[0]
	leftmark = 1
	rightmark = len(unsorted_array)-1
	while leftmark <= len(unsorted_array) -1 and rightmark >=1:
		while unsorted_array[leftmark] <= pivot_value:
			leftmark += 1
			if leftmark == len(unsorted_array) -1: break
		
		while unsorted_array[rightmark] >= pivot_value:
			rightmark -= 1
			if rightmark == 1: break
			
		if leftmark >= rightmark:
			break
			
		unsorted_array[leftmark], unsorted_array[rightmark]	= unsorted_array[rightmark], unsorted_array[leftmark]
		
	if leftmark == rightmark == 1:
		unsorted_array[1:] = quick_sort(unsorted_array[1:]) #sort and replace #slicing is a copy, not original.
	else:
		unsorted_array[0], unsorted_array[rightmark] = unsorted_array[rightmark], unsorted_array[0]
		unsorted_array[:rightmark] = quick_sort(unsorted_array[:rightmark]) #sort and repalce
		if rightmark < len(unsorted_array) -1: 
			unsorted_array[rightmark+1:] = quick_sort(unsorted_array[rightmark+1:]) #sort and replace
	
	return unsorted_array


print (quick_sort([2, 0, 3, -8, -1, 20, 70, -100, 0, 100, 70, 29]))



#======= 19 =======
#This is to practice merge sort without class
def merge_sort (unsorted_array):
	if len(unsorted_array) <= 1:
		return	unsorted_array
		
	ind_mid = len(unsorted_array)//2
	left_half = unsorted_array[:ind_mid]
	right_half = unsorted_array[ind_mid:]
	
	sorted_left = merge_sort(left_half)
	sorted_right = merge_sort(right_half)
	
	return merge(sorted_left, sorted_right)

def merge(sorted_left, sorted_right):
	coll_ele = []
	while len(sorted_left) > 0 and len(sorted_right) > 0:
		if sorted_left[0] < sorted_right[0]:
			coll_ele.append(sorted_left[0])
			sorted_left.remove(sorted_left[0])
		else:
			coll_ele.append(sorted_right[0])
			sorted_right.remove(sorted_right[0])

	if sorted_left != []:
		coll_ele += sorted_left
	elif sorted_right != []:
		coll_ele += sorted_right
	
	return coll_ele

print (merge_sort([2, 3, -3, 0, -99, 100]))




#===== 18 =======
#This is to practice merge sort with class
class MergeSort:
	def __init__(self, array):
		self.array = array
		self.length = len(array)
		
	def merge_sort(self):
		if self.length <= 1:
			return MergeSort(self.array)
			
		ind_mid = self.length//2
		left_half = MergeSort(self.array[:ind_mid])
		right_half = MergeSort(self.array[ind_mid:])
		
		left_half = left_half.merge_sort()
		right_half = right_half.merge_sort()
		
		return left_half.merge(right_half)
		
	def merge(self, the_other_half):
		coll_ele = []
		while self.length > 0 and the_other_half.length > 0:
			if self.array[0] < the_other_half.array[0]:
				coll_ele.append(the_other_half.array[0])
				the_other_half.array.remove(the_other_half.array[0])
				the_other_half.length = len(the_other_half.array)
			else:
				coll_ele.append(self.array[0])
				self.array.remove(self.array[0])
				self.length = len(self.array)
		
		if self.length == 0:
			coll_ele += the_other_half.array
		else:
			coll_ele += self.array
		
		return	MergeSort(coll_ele)
		
		
my_merge_sort = MergeSort([2,3, 0, -2, 3, 9]).merge_sort()
print (my_merge_sort.array)
		
		


#========= 17 =========
#This is to practice insert an element into a sorted array
class BinaryInsert:
	def __init__(self, array):
		self.array = array.copy()
		self.length = len(self.array)
		if self.array[0] < self.array[-1]:
			self.ascend = 1
			self.min_ele = self.array[0]
			self.max_ele = self.array[-1]
		elif self.array[0] > self.array[-1]:
			self.ascend = 0
			self.min_ele = self.array[-1]
			self.max_ele = self.array[0]
		else:
			self.ascend = None
			print ('Error: the initial and last elements are the same.')

	def binary_insert(self, value):
		if self.ascend == 1:
			if value > self.max_ele:
				self.array.append(value)
				print ('new array {}'.format(self.array))
				return True
			elif value < self.min_ele:
				self.array.insert(0, value)
				print ('new array {}'.format(self.array))
				return True
			#deal with the situation where value is within the array range
			ind_start = 0
			ind_end = self.length-1
			ind_mid = (ind_start+ind_end)//2
			
			while ind_start < ind_end-1:
				if self.array[ind_mid] < value:
					ind_start = ind_mid
				else:
					ind_end = ind_mid	
				ind_mid = (ind_start+ind_end)//2
			self.array.append('')
			for i in range(self.length, ind_end, -1):
				self.array[i] = self.array[i-1]
			self.array[ind_end] = value
			print ('new array {}'.format(self.array))

		elif self.ascend == 0:
			if self.min_ele > value:
				self.array.append(value)
				print ('new array {}'.format(self.array))
				return True
			elif self.max_ele < value:
				self.array.insert(0, value)
				print ('new array {}'.format(self.array))
				return True
				
			ind_start = 0
			ind_end = self.length-1
			
			while ind_start < ind_end -1:
				ind_mid = (ind_start + ind_end)//2
				if self.array[ind_mid] < value:
					ind_end = ind_mid
				else:
					ind_start = ind_mid
			#since we find where to insert such an element, we can determine whether to insert it from the right or the left end. We just use the right end
			self.array.append('')
			for i in range(self.length, ind_end, -1):
				self.array[i] = self.array[i-1]
			self.array[ind_end] = value
			print ('new array {}'.format(self.array))
			
		else:
			print ('Error: the initial and last elements are the same.')
			return False

BinaryInsert([2,3,4,7,9,10]).binary_insert(3)

BinaryInsert([10, 8, 7, 6, 2, 0]).binary_insert(117)


#======= 16 ======
#This script is to practice binary search for a sorted list
class BinarySearch:
	def __init__(self, array):
		self.array = array.copy()
		self.length = len(self.array)
		if self.array[0] < self.array[-1]:
			self.ascend = 1
			self.min_ele = self.array[0]
			self.max_ele = self.array[-1]
		elif self.array[0] > self.array[-1]:
			self.ascend = 0
			self.min_ele = self.array[-1]
			self.max_ele = self.array[0]
		else:
			self.ascend = None
			print ("Error: the first and last elements are the same.")
	
	def binary_search(self, value):
		ind_start = 0
		ind_end = self.length-1
		
		if value < self.min_ele or value > self.max_ele:
			print (str(value)+' is not found')
			return False
		
		if self.ascend == 1: #array in ascending order
			while ind_start <= ind_end:
				ind_mid = (ind_start+ind_end)//2
				if self.array[ind_mid] == value:
					print (str(value)+' is found')
					return True
				elif self.array[ind_mid] > value:
					ind_end = ind_mid -1
				elif self.array[ind_mid] < value:
					ind_start = ind_mid+1
			
			print(str(value)+' is not found.')
					
					
		elif self.ascend == 0: #array in descending order
			while ind_start <= ind_end:
				ind_mid = (ind_start+ind_end)//2
				if self.array[ind_mid] == value:
					print (str(value)+' is found')
					return True
				elif self.array[ind_mid] > value:
					ind_start = ind_mid+1
				elif self.array[ind_mid] < value:
					ind_end = ind_mid -1
					
			print(str(value)+' is not found.')
		else:
			print ('Wrong')
			
		
my_binary_search = BinarySearch([1,2,3, 100, 123])
my_binary_search.binary_search(-1123)



#========== 15 ========
#This is to practice insertion sort
class InsertionSort:
	def __init__(self, array):
		self.array = array
		self.length = len(array)
		
	def insertion_sort(self):
		for ind_out in range(1, self.length):
			temp = self.array[ind_out]
			j = ind_out - 1
			while self.array[j] < temp and j>=0:
				self.array[j+1] = self.array[j] #move the lower element to upper index. Don't use swap to save operations.
				j -= 1
			j+=1
			self.array[j] = temp 
		print (self.array)


InsertionSort([7, 0, 100, -11, 70, -1]).insertion_sort()


#======== 14 =========
#This is to practice selection sort algorithm
class SelectionSort:
	def __init__(self, array):
		self.array = array.copy()
		self.length = len(array)
	
	def selection_sort(self):
		for ind_out in range(self.length-1):
			large_ind = ind_out
			for ind_ins in range(ind_out+1, self.length):
				if self.array[large_ind] < self.array[ind_ins]:
					large_ind = ind_ins
			self.array[ind_out], self.array[large_ind] = self.array[large_ind], self.array[ind_out]

		print (self.array)


SelectionSort([2,3,-7, 0, -1, 100]).selection_sort()



#========== 13 ==========
#This is to practice bubble sort
class BubbleSort:
	def __init__(self, array):
		self.array = array
		self.length = len(array)
		print('test')

	#def bubblesort(self):
		for ind_out in range(self.length-1, 0, -1): #control the length
			for ind_ins in range(ind_out):
				if self.array[ind_ins] < self.array[ind_ins+1]: #compare with the adjacent elements
					self.array[ind_ins], self.array[ind_ins+1] = self.array[ind_ins+1], self.array[ind_ins] #swap is they are in the wrong order
		print (self.array)

BubbleSort([2,3,4, -1, 0, 100])#.bubblesort()



#========== 12 =========
#Use pointers to implement the min heap
#turns out we need to use doubly linked lists
#turns out it is hard to implement heap using pointers. No complete rule determining where 
#to put the new nodes. Thus use array is the easiest way to implement heap structure.
class Node:
	def __init__(self, value = None):
		self.value = value
		
		#for each node, we need one link to previous node and two next links to the next left and right nodes.
		self.prev_node = None
		self.next_node_left = None
		self.next_node_right = None
	
	def collect_nodes(self):
		nodes = []
		if self is not None:
			if self.next_node_left is not None:
				self.next_node_left.collect_nodes()
				
		
	def build_raw_tree(self, value=None):
		if self.value is None:
			self.value = value
		else:
			if self.next_node_left is None:
				self.next_node_left = Node(value)
				self.next_node_left.prev_node = self
			elif self.next_node_right is None:
				self.next_node_right = Node(value)
				self.next_node_right.prev_node = self
			else:
				#collect all the existing non-root nodes
				
				
				
				for parent in [self.next_node_left, self.next_node_right]:
					parent.build_raw_tree(value)
				
		

#======== 11 ========
#This is to practice heap using arrays as implementation
class MinHeap:
	def __init__(self, array):
		self.array = array.copy()
		self.max_index = len(array)-1
		self.max_parent = (len(array)-2)//2
		print (self.max_index, self.max_parent)
		
	def heapify(self, index): #deal with each parent node thoroughly
		if index > self.max_parent:
			return False
		#find the minimum
		min_ind = index
		if self.array[min_ind] > self.array[2*index+1]: #the parent node has at least one child
			min_ind = 2*index+1
		if 2*index+2 <= self.max_index and self.array[min_ind] > self.array[2*index+2]:
			min_ind = 2*index+2
		
		if min_ind != index: #found new minimum
			self.array[index], self.array[min_ind] = self.array[min_ind], self.array[index] #switch object
			#will this switch change the propety of node(min_ind)?
			self.heapify(min_ind) #deal with the above question
	
	def build_min_heap(self):
		#bottom up to get the minimum heap
		for parent in range(self.max_parent, -1, -1):
			self.heapify(parent)
	
	def heapop(self):
		#switch the root node and the last leaf, in order to use list.pop() to get and remove the minimum value and apply heapify to the updated root node.
		self.array[0], self.array[-1] = self.array[-1], self.array[0]
		min_value = self.array.pop()
		#since the array is modified, don't forget to update the information of new array.
		self.max_index = len(self.array)-1
		self.max_parent = (len(self.array)-2)//2
		self.heapify(0) #since only the root node is modified, only apply heapify on the root node.
		
		return min_value
		
array = [2,3,0,-88,23.0]	
array = ['I', 'Love', 'You', '!', 'Can', 'You', 'Marry', 'Me', '?', 'Lol']

my_min_heap = MinHeap(array)	
my_min_heap.build_min_heap()
print (my_min_heap.array[0])	

for i in range(len(array)):
	print (my_min_heap.heapop(), end =' ')		
	


#======== 10 ========
#This is to practice binary trees
class Node:
	def __init__(self, value = None):
		self.value = value
		self.left = None
		self.right = None
	
	def grow(self, value = None):
		if self.value is None: #if head node is not initialized, then take care of it.
			self.value = value
		else:
			if value < self.value: #smaller than the current node
				if self.left is None:
					self.left = Node(value)
				else: #existing node, repeat this process
					self.left.grow(value)
			else: #no smaller than the current, i.e. >=
				if self.right is None:
					self.right = Node(value)
				else:#existing node, repeat this process
					self.right.grow(value)
	
	def print_tree(self):
		if self.left is not None:
			self.left.print_tree()
		print ('{} '.format(self.value), end = ' ')
		if self.right is not None:
			self.right.print_tree()	
	
	def find_value (self, value)	: #exhaust all the possibilities
		if self.value == value:
			print ('{} exists in the binary tree.'.format(value))	
			return True
		if self.right is not None:
			#print ('right')
			self.right.find_value(value)
		if self.left is not None:
			#print ('left', str(self.left.value))
			self.left.find_value(value)
		print (False)
	
	def find_value_revised(self, value): #utilize the properties of binary tree left<=parent<=right
		if value < self.value:
			if self.left is None:
				print ('{} does not exist'.format(value))
			else:
				self.left.find_value_revised(value)
		elif value > self.value:
			if self.right is None:
				print ('{} does not exist'.format(value))
			else:
				self.right.find_value_revised(value)
		else:
			print (('{} does exist'.format(value)))
		
	def inorder_trav(self): #left -> root -> right
		col_ele = []
		if self is not None:
			if self.left is not None:
				col_ele = self.left.inorder_trav()
			col_ele.append(self.value)
			if self.right is not None:
				col_ele.extend(self.right.inorder_trav())
		return col_ele
	
	def preorder_trav(self): #root -> left -> right
		col_ele = []
		if self is not None:
			col_ele.append(self.value)
			if self.left is not None:
				col_ele.extend(self.left.preorder_trav())
			if self.right is not None:
				col_ele.extend(self.right.preorder_trav())
		return col_ele
		
	def postorder_trav(self): #left -> right -> root
		col_ele = []
		if self is not None:
			if self.left is not None:
				col_ele.extend(self.left.postorder_trav())
			if self.right is not None:
				col_ele.extend(self.right.postorder_trav())
			col_ele.append(self.value)
		return col_ele	
				
		
my_binary_tree = Node()

for i in [27, 14, 35, 10, 19, 31,42]:
	my_binary_tree.grow(i)

my_binary_tree.print_tree()

for i in[8, 2, -1, 0, 10, 30, -7, 100]:
	#my_binary_tree.find_value_revised(i)
	my_binary_tree.find_value(i)

#my_binary_tree.find_value_revised(232)
print ('====')
my_binary_tree.find_value(232)

print (my_binary_tree.inorder_trav())
print (my_binary_tree.preorder_trav())
print (my_binary_tree.postorder_trav())


#======= 9 =======
#This is to practice queue (FIFO/LILO). It does not matter how you store the items, it is how
#you retrieve the elements
class Queue:
	def __init__(self):
		self.queue = []

	def add_ele_insert (self, ele = None):
		self.queue.insert(0, ele)
		
	def get_ele_insert (self):
		return self.queue.pop() if len(self.queue) > 0 else "No more elements in the queue"
	
	def add_ele_append (self, ele=None):
		self.queue.append(ele)
	
	def get_ele_append (self):
		try:
			return self.queue.pop(0)
		except IndexError:
			print('IndexError')
		except:
			print('Something else went wrong')
		else:
			print('good') #meaningless because if no error, the control will return before getting to this point.

my_queue = Queue()

print('Using insert')
for i in range(10):
	my_queue.add_ele_insert(i)

for i in range(10):
	print (my_queue.get_ele_insert())

print('Using append')
for i in range(10):
	my_queue.add_ele_append(i)

for i in range(10):
	print (my_queue.get_ele_append())


#======== 8 =========
#This is to practice stack (FILO/LIFO)
class Stack:
	def __init__(self):
		self.stack = []
	
	def add_ele(self, value=None):
		self.stack.append(value)
	
	def get_ele(self): #pop() by default removes the last element
		return self.stack.pop(-1) if len(self.stack)>0 else "No more elements in the stack."

my_stack = Stack()
for i in range(10):
	my_stack.add_ele(i)

for i in range(12):
	print (my_stack.get_ele())



#======== 7 ========
#This is to practice doubly linked list
class Node:
	def __init__(self, seniority=None, age=None):
		self.seniority = seniority
		self.age = age
		self.prev_node = None
		self.next_node = None
		
class DoubleLinkedList:
	def __init__(self):
		self.head_node = None
	
	def build(self, seniority=None, age=None):
		new_node = Node(seniority, age)
		if self.head_node is None:
			self.head_node = new_node
		else:
			self.head_node.next_node = new_node
			temp = self.head_node
			self.head_node = new_node
			self.head_node.prev_node = temp
	
	def print_dll(self): #everytime it is called, self.head_node goes back to the head node.
		while self.head_node is not None:
			print (self.head_node.seniority, self.head_node.age)
			self.head_node = self.head_node.prev_node
			
my_family = DoubleLinkedList()
my_family.build('GrandFather', 100)
#my_family.print_dll()
my_family.build('Father', 80)
my_family.print_dll()
my_family.build('Son', 60)
my_family.print_dll()



#======== 6 =========
#This is to practice singly linked list in a different way
class Node:
	def __init__(self, seniority=None, age=None):
		self.seniority = seniority
		self.age = age
		self.next_node = None

class SinglyLinkedList:
	def __init__(self):
		self.head_node = None
	
	def build(self, seniority=None, age=None):
		new_node = Node(seniority, age)
		if self.head_node is None:
			self.head_node = new_node
		else:
			self.head_node.next_node = new_node
			self.head_node = new_node
	
	def print_sll(self): #Since we lost track of the previous nodes, we can not go back 
						 #with the singly linked list. Use doubly linked list.
		print (self.head_node.seniority, self.head_node.age)
	
my_family = SinglyLinkedList()
my_family.build('GrandFather', 100)
my_family.print_sll()
my_family.build('Father', 80)
my_family.print_sll()
my_family.build('Son', 60)
my_family.print_sll()




#======= 5 =======
#This is to practice singly linked list (one direction)
class Node:
	def __init__(self, seniority=None, age = None):
		self.seniority = seniority
		self.age = age
		self.next_node = None
		
class SinglyLinkedList:
	def __init__(self):
		self.head_node = None #head node, has nothing but structure
	
	def print_sll(self): 
		node = self.head_node
		while node is not None:
			print (node.seniority, node.age)
			node = node.next_node
	
	
my_family = SinglyLinkedList()
grand_father = Node('GrandFather', 100)
father = Node('Father', 70)
son = Node('Son', 50)

my_family.head_node = grand_father
grand_father.next_node = father
father.next_node = son

my_family.print_sll()


#======= 4 =======
#This is to practice ChainMap
import collections

dict_1 = {2:3, 'love': 'dong'}
dict_2 = {2: 7, 'love': 'sy'}
combine_dict = collections.ChainMap(dict_1, dict_2) #like stack. FILO

print ('Map:', combine_dict.maps)
print ('Keys:', list(combine_dict.keys()))
print ('Values:', list(combine_dict.values()))

for k, v in combine_dict.items():
	print (k, v)

#change dict_1
dict_1['test'] = 'test'
print (combine_dict) #chainmap reflects the changes made in individual dictionary



#======== 3 =========
#This is to practice matrix (numpy array)
import numpy as np

a = np.array([[2,3], ['love', 'you']])

for row in a:
	for column in row:
		print (column)

b = a.reshape((-1, 1))
print (b)

c = np.append(a, [[3,4]], 0) #append new rows
print ('c', c)

d = np.insert(c, [0, 1, -1], [[0], [0], [0]], 1) #insert columns at positions [0, 1, -1]
print(d)



#========= 2 ========
#This is to practice 2D array (list)
a = [[2,3], ['love', 'me']]
for row in a:
	for column in row:
		print (column)


#====== 1 =========
#This is to practice arrays, which are very similar to lists, though arrays only contain 
#the same-typle objects, while lists do not have such constraints.
import array as arr
a = arr.array('i', [2, 3, 3, 4]) #can ONLY same-type objects. same-type class object?
b = arr.array('d', [2, 3.]) #small scope can be converted to large scope types. does not loss information

for i in a:
	print (i, 'index:', a.index(i)) 

for i in range(len(b)):
	print(b[i], 'index:', b.index(b[i])) #You will find 2 is converted as 2.0 -> scope conversion

a.insert(1, 23)
print ('after being inserted at index 1:', a)
del(a[1])
print ('after deleting the element at 1:', a)
a.remove(3)
print ('after removing the first occurance of 3:', a)

print (a+a[1:]+2*a) #2*a == a+a

for i in range(len(a)):
	a[i] = 2.1*a[i] #must be the same type, otherwise error, or can be lower scope type
	
"""

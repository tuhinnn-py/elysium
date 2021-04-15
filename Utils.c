/*********
	1. prtarr - Print a 1D array
	2. display - Print a 2D array
	3. topSortKahn - Topsort using Kahn's algorithm
	4. detectCycle - Detect cycles using Kahn's algorithm
	5. topSort_DFS - Topsorting algorithm using DFS :: utilty function
	6. topSort - Topsorting algorithm using DFS 
	7. swap - Dijkstra's algorithm using binary heaps :: utility
	8. getNode - Dijkstra's algorithm using binary heaps :: utility
	9. heapifyPair - Dijkstra's algorithm using binary heaps :: utility
	10. dijkstraHeap - Dijkstra's algorithm using binary heaps :: utility
	11. a2dp - Convert a 2D array to a double pointer
	12. addDummy - Johnson's algorithm :: utility
	13. findMin - Dijkstra's algorithm using linear search :: utility
	14. dijkstraLinear - Dijkstra's algorithm using linear search 
	15. min - Find minimum of two numbers
	16. bellmanFord - Bellman-Ford algorithm
	17. johnson - Johnson's algorithm
	18. fillArray - Fill a 2D array with a given value
	19. createSparseTable - Create a sparse table for range queries
	20. rangeQueryOverlap - Range queries for overlapping associative functions 
	21. rangeQueryCascading - Range queries for non-overlapping associative functions
	22. addChild - Add a child to a node :: utility
	23. buildTree - rootTree :: utility
	24. rootTree - Root a tree at a given index :: utility
	25. createTree - Convert a 2D array to a tree by finding it's centers
	26. eulerianTour - Create an eulerian tour from a given tree
	27. cmpfunc - A comparator function for qsort :: utility
	28. findEncodingUtils - Generate a AHU Encoding for a given tree taking a 2D array as input :: utils
	29. findEncoding - Generate a AHU Encoding for a given tree taking a 2D array as input
	30. lca - Calculate the lowest common ancestor using Eulerian tour and range minimum query
	31. revAdjacency - Reverse a given adjacency matrix :: utility
	32. kosarajuUtility - Kosaraju's algorithm for finding strongly connected components :: utility
	33. scckosaraju - Kosaraju's algorithm for finding strongly connected components
	34. tarjandfs - Tarjan's algorithm for finding strongly connected components :: utility
	35. tarjanUtility - Tarjan's algorithm for finding strongly connected components :: utility
	36. sccTarjan - Tarjan's algorithm for finding stongly connected components
	37. find - Find method for Union-Find data structure
	38. uni - Union method for Union-Find data structure
	39. lbit - Calculate the value of the least significant bit
	40. calpfx - Calculate the prefix sum from a fenwick tree :: utility
	41. pointUpdate - Point Update method for a fenwick tree :: utility
	42. rangeCascading - Calculate the 'sum' range query from a fenwick tree :: utility
	43. createFTree - Construct a Fenwick tree for a given array
	44. max - Calculate the maximum of two numbers
	45. findDepth - find the depth of a tree :: utility
	46. createParentTable - Create parent table for a given tree for Binary Uplifting algorithm :: utility
	47. binaryLifting - Binary Lifting algorithm for finding the lowest common ancestor of two given nodes 
	48. fillSegmentTree - Fill a segment Tree recursively :: utility
	49. createSegmentTree - create a segment tree from a given array
	51. intersects - Check if two bounds intersect each other :: utility
	52. sumQuery - Find the range sum query in a segment tree :: utility
	53. updateQuery - Update an index of an array and reflect that in the segment tree
	54. rangeQuerySegment - Find the range sum query in a segment tree 
	55. fordFulklersonDFS - Find all augmenting paths using DFS in a residual graph coupled with a delta for capacity scaling :: utility
	56. findDelta - Find the value of delta for capacity scaling
	57. fordFulklerson - Find the maximum flow using Ford Fulkerson algorithm
	58. edmondsKarpBFS - Find all augmenting paths using BFS in a residual graph coupled with a delta for capacity scaling :: utility
	59. edmondsKarp - Find the maximum flow using Edmonds Karp algorithm
	60. dinicDFS - Find an augmenting path in a level graph :: utility
	61. dinicBFS - create a level graph coupled with capacity scaling:: utility
	62. dinic - Find the maximum flow using Dinic's algorithm
	63. findNext - Find the next node with an excess flow (if there exists one) :: utility
	64. pushRelabel - Find the maximum flow with the Push-Relabel algorithm
	65. augmentMatching - Augment the current matching with the augmenting path provided in Hopcroft Karp's algorithm:: utility
	66. createLevelGraph - Create an alternating Breadth-First-Search tree :: utility
	67. findPath - find an augmenting path from the BFS tree :: utility
	68. hopcroftKarp - Return a maximum matching of a given bipartite graph
	69. comparator - comparator function for comparing two nodes in descending order according to it's degrees :: utility
	70. welshPowell - Return the chromatic number of an undirected graph given the graph in the form of an adjacency matrix
	71. getTrieNode - Initialise and return a trie node : utility
	72. setFailureLink - Initialise the failure link of a trie node with the given link :: utility
	73. setDictionaryLink - Initialise the dictionary link of a trie node with the given link :: utility
	74. setEnd - Initialise the is_end flag of a trie node to decalre it as a vocabulary word:: utility
	75. setWord - Initialise the vocab word of a trie node whose is_end flag is true:: utility
	76. hasLink - Returns if a given trie node has a child at a particular position :: utility
	77. substr - Implements the substring function of C++ :: utility
	78. addVocab - Adds the given vocabulary word to the trie tree :: utility
	79. getWord - Returns the node containing the end character of the given word if it exists and null if it doesn't :: utility
	80. hasWord - Checks if the trie has the given word :: utility
	81. addFailureLinks - Adds failure links to the given trie tree :: utility
	82. addDictionaryLinks - Adds dictionary links to the given trie tree :: utility
	83. create_trie_aho_corasick - Creates the AHO - CORASICK automaton 
	84. getHashCode - Returns the hash value of a string :: utility
	85.search_for_words - Searches the AHO - CORASICK automaton for the occurance of vocabulary words
	86. knuth_morris_pratt - Returns the indices of the text string where the pattern is found using the Knutt-Morris-Pratt algorithm
	87. calculate_z_value - Returns an array containing z-values of a string 
	88. z_algorithm - Returns the indices of the text string where the pattern is found using the Z-algorithm
	89. main - Main!!
**********/

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdbool.h"
#include "time.h"
#include "math.h"

#define N 100
#define INF 9999999
#define and &&
#define or ||
#define none NULL

struct Stack{
	int arr[N];
	int top;
};
typedef struct Stack S;

struct queue{
	int arr[N];
	int front, rear;
};
typedef struct queue q;

bool isEmpty(q* que)
{
	return (que->front > que->rear or que->front == -1)?true:false;
}

void push(q* que, int elem)
{
	if(que->rear == -1)
		que->front = 0;
	que->arr[++que->rear] = elem;
}

int dequeue(q* que)
{
	return que->arr[que->front++];
}

struct p{
	int dist;
	int node;
};
typedef struct p pair;

void prtarr(int* arr, int V)
{
	int j;
	for(j = 0; j < V; j++)
		printf("%d ", arr[j]);
	printf("\n");
}

void display(int** arr, int V)
{
	int i, j;
	for(i = 0; i < V; i++)
	{
		for(j = 0; j < V; j++)
			printf("%d ", arr[i][j]);
		printf("\n");
	}
}

int* topSortKahn(int** adj, int V)
{
	int i,j;
	int in[V];
	int* ordering;
	ordering = (int*)malloc(sizeof(int) * V);

	memset(in, 0, sizeof(in));
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(adj[i][j] < INF and adj[i][j] > 0)
				in[j] += 1;
	
	q* que = (q*)malloc(sizeof(q));
	que->front = -1;
	que->rear = -1;
	
	for(i = 0; i < V; i++)
		if(in[i] == 0)
			push(que, i);
	
	j = 0;
	
	while(!isEmpty(que))
	{
		int node = dequeue(que);
		ordering[j++] = node;
		
		for(i = 0; i < V; i++)
			if(adj[node][i] < INF and adj[node][i] > 0)
			{
				in[i] -= 1;
				if(in[i] == 0)
					push(que, i);
			}	
	}
	
	return ordering;
}

bool detectCycle(int** adj, int V)
{
	int i,j;
	int in[V];
	memset(in, 0, sizeof(in));
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(adj[i][j] < INF and adj[i][j] > 0)
				in[j] += 1;
	
	q* que = (q*)malloc(sizeof(q));
	que->front = -1;
	que->rear = -1;
	
	for(i = 0; i < V; i++)
		if(in[i] == 0)
			push(que, i);
	
	j = 0;
	
	while(!isEmpty(que))
	{
		int node = dequeue(que);
		j++;
		
		for(i = 0; i < V; i++)
			if(adj[node][i] < INF and adj[node][i] > 0)
			{
				in[i] -= 1;
				if(in[i] == 0)
					push(que, i);
			}	
	}

	return !(j == V);
}

void topSort_DFS(int** adj, bool* visited, int* ordering, int* idx, int node, int V)
{
	int i;
	visited[node] = true;
	
	for(i = 0; i < V; i++)
		if(adj[node][i] < INF and !visited[i])
			topSort_DFS(adj, visited, ordering, idx, i, V);
			
	ordering[*idx] = node;
	*idx -= 1;
}

int* topSort(int** adj, int V)
{
	int i;
	int idx = V - 1;
	
	bool* visited;
	visited = (bool*)malloc(sizeof(bool) * V);
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	int* ordering;
	ordering = (int*)malloc(sizeof(int) * V);

	while(idx > -1)
	{
		int node = rand() % V;
		if(!visited[node])
			topSort_DFS(adj, visited, ordering, &idx, node, V);	
	}

	return ordering;
}

void swap(pair* x, pair* y)
{
	pair t;
	t = *x;
	*x = *y;
	*y = t;
}

pair* getNode(int distance, int location)
{
	pair* xy = (pair*)malloc(sizeof(pair));
	xy->dist = distance;
	xy->node = location;
	return xy;
}

void heapifyPair(pair* arr[N], bool visited[N], int V, int i)
{
	int left = 2 * i + 1;
	int right = 2 * i + 2;
	
	if(left < V)
		heapifyPair(arr, visited, V, left);
	if(right < V)
		heapifyPair(arr, visited, V, right);
	
	if(left < V and !visited[arr[left]->node] and arr[left]->dist < arr[i]->dist)
		swap(arr[left], arr[i]);
	if(right < V and !visited[arr[right]->node] and arr[right]->dist < arr[i]->dist)
		swap(arr[right], arr[i]);
	
}

int* dijkstraHeap(int** adj, int V, int source)
{
	int i, counter = 0, top = -1;
	bool* visited;
	visited = (bool*)malloc(sizeof(bool)*V);
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	pair* arr[N];
	
	int* distance;
	distance = (int*)malloc(sizeof(int)*V);
	for(i = 0; i < V; i++)
		distance[i] = INF;
		
	distance[source] = 0;
	for(i = 0; i < V; i++)
		arr[++top] = getNode(distance[i], i);
	
	while(counter < V)
	{
		heapifyPair(arr, visited, top + 1, counter);
		int location = arr[counter]->node;
		visited[location] = true;
		
		for(i = 0; i < V; i++)
		{
			if(adj[location][i] < INF and distance[i] > distance[location] + adj[location][i])
			{
				distance[i] = distance[location] + adj[location][i];
				arr[++top] = getNode(distance[i], i);
			}
		}
		counter++;
	}
	
	return distance;
}

int** a2dp(int adj[N][N], int V)
{
	int i, j;
	int** adj_;
	adj_ = (int**)malloc(sizeof(int*) * V);
	
	for(i = 0; i < V; i++)
		adj_[i] = (int*)malloc(sizeof(int) * V);
	
	for(i = 0; i < V; i++)
	{
		for(j = 0; j < V; j++)
		{
			adj_[i][j] = adj[i][j];
		}
	}
	
	return adj_;
}

int** addDummy(int** adj, int n)
{
	int i, j;
	int** adj_;
	adj_ = (int**)malloc(sizeof(int*) * (n + 1));
	
	for(i = 0; i < n + 1; i++)
		adj_[i] = (int*)malloc(sizeof(int) * (n + 1));
	
	for(i = 0; i < n + 1; i++)
		adj_[0][i] = 0;
		
	for(i = 1; i < n + 1; i++)
		adj_[i][0] = INF;
	
	for(i = 1; i < n + 1; i++)
		for(j = 1; j < n + 1; j++)
			adj_[i][j] = adj[i -1][j - 1];
	
	return adj_;
}

int findMin(int* arr, bool* visited, int n)
{
	int m = INF;
	int i, location;
	for(i = 0; i < n; i++)
	{
		if(!visited[i] and m > arr[i])
		{
			m = arr[i];
			location = i;
		}
	}
	return location;
}

int* dijkstraLinear(int** adj, int V, int source)
{
	int i, counter = 0;
	bool* visited;
	visited = (bool*)malloc(sizeof(bool)*V);
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	int* distance;
	distance = (int*)malloc(sizeof(int)*V);
	for(i = 0; i < V; i++)
		distance[i] = INF;
	
	distance[source] = 0;
	
	while(counter < V)
	{
		int location = findMin(distance, visited, V);
		visited[location] = true;
		
		for(i = 0; i < V; i++)
		{
			if(adj[location][i] < INF and distance[i] > distance[location] + adj[location][i])
				distance[i] = distance[location] + adj[location][i];
		}
		counter++;
	}
	
	return distance;
}

int min(int a, int b)
{
	return (a>b)?b:a;
}

int* bellmanFord(int** adj, int V, int source)
{
	int i, j, k;
	
	int* distance;
	distance = (int*)malloc(sizeof(int)*V);
	for(i = 0; i < V; i++)
		distance[i] = INF;
		
	distance[source] = 0;

	for(i = 1; i < V; i++)
	{
		for(j = 0; j < V; j++)
		{
			int  minDist = INF;
			for(k = 0; k < V; k++)
				if(adj[k][j] < INF)
					minDist = min(minDist, distance[k] + adj[k][j]);
			distance[j] = min(distance[j], minDist);
		}
	}
	
	return distance;
}

int** johnson(int** adj, int V)
{
	int i, j;
	int** adj_;
	
	int** apsp;
	apsp =(int**)malloc(sizeof(int*) * V);
	
	for(i = 0; i < V; i++)
		apsp[i] = (int*)malloc(sizeof(int) * V);
	
	adj_ = addDummy(adj, V);
	
	int* distance;
	distance = bellmanFord(adj_, V + 1, 0);
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(adj[i][j] < INF)
				adj[i][j] += distance[i + 1] - distance[j + 1];
	
	for(i = 0; i < V; i++)
		apsp[i] = dijkstraHeap(adj, V, i);
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(apsp[i][j] < INF)
				apsp[i][j] += distance[j + 1] - distance[i + 1];
	
	display(apsp, V);
	
	return apsp;
}

void fillArray(int** arr, int R, int C, int fill)
{
	int i,j;
	for(i = 0; i < R; i++)
		for(j = 0; j < C; j++)
			arr[i][j] = fill;
}

struct sparse_index{
	int** stbl;
	int** idxtbl;
};
typedef struct sparse_index sidx;

sidx* createSparseTable(int* arr, int S, int (*func)(int, int))
{
	int i, j;
	int p = log(S) / log(2);
	
	int** stbl;
	stbl = (int**)malloc(sizeof(int*) * (p + 1));
	
	for(i = 0; i <= p; i++)
		stbl[i] = (int*)malloc(sizeof(int) * S);
	
	int** idxtbl;
	idxtbl = (int**)malloc(sizeof(int*) * (p + 1));
	
	for(i = 0; i <= p; i++)
		idxtbl[i] = (int*)malloc(sizeof(int) * S);
	
	fillArray(stbl, p + 1, S, INF);
	fillArray(idxtbl, p + 1, S, -1);
	
	for(i = 0; i < S; i++)
	{
		stbl[0][i] = arr[i];
		idxtbl[0][i] = i;
	}
	
	for(i = 1; i <= p; i++)
		for(j = 0; j + (1 << i) <= S; j++)
		{
			int left = stbl[i-1][j];
			int right = stbl[i -1][j + (1 << (i - 1))];
			
			stbl[i][j] = func(left, right); 
			
			if(left <= right)
				idxtbl[i][j] = idxtbl[i-1][j];
			else
				idxtbl[i][j] = idxtbl[i -1][j + (1 << (i - 1))];
		}
	
	sidx* sdx = (sidx*)malloc(sizeof(sidx));
	sdx->idxtbl = idxtbl;
	sdx->stbl = stbl;
	
	return sdx;
}

int rangeQueryOverlap(int* arr, int S, int (*func)(int, int), int lower, int upper, bool getIndex)
{
	int index;
	int range = upper - lower + 1;
	int p = log(range) / log(2);
	
	int** stbl;
	int** idxtbl;
	
	sidx* sdx = createSparseTable(arr, S, func);
	stbl = sdx->stbl;
	idxtbl = sdx->idxtbl;
	
	int left = stbl[p][lower];
	int right = stbl[p][upper + 1 - (1 << p)];
	
	if(left <= right)
		index = idxtbl[p][lower];
	else
		index = idxtbl[p][upper + 1 - (1 << p)];
	
	return !getIndex? func(left, right): index;
}

int rangeQueryCascading(int* arr, int S, int (*func)(int, int), int lower, int upper)
{
	int i;
	int range = upper - lower + 1;
	int p = log(range) / log(2);
	
	int** stbl;
	int** idxtbl;
	
	sidx* sdx = createSparseTable(arr, S, func);
	stbl = sdx->stbl;
	idxtbl = sdx->idxtbl;
	
	int res = stbl[p][lower];
	int last = lower + (1 << p);
	
	for(i = p - 1; i >= 0; i--)
		if(range & (1 << i))
		{ 
			res = func(res, stbl[i][last]);
			last += (1 << i);
		}
	
	return res;
}

struct treeNode{
	int idx;
	struct treeNode* parent;
	struct treeNode* children[N];
	int child;
	char str[N]; 
};
typedef struct treeNode tn;

void addChild(tn* node, int idx)
{
	tn* n = (tn*)malloc(sizeof(tn));
	n->idx = idx;
	n->child = 0;
	n->parent = node;
	node->children[node->child++] = n;
}

void buildTree(int** adj, int V, tn* node)
{
	int i;
	
	for(i = 0; i < V; i++)
		if(adj[node->idx][i])
		{
			if(node->parent == NULL or i != node->parent->idx)
			{
				addChild(node, i);
				buildTree(adj, V, node->children[node->child - 1]);
			}
		}
}

tn* rootTree(int** adj, int V, int node)
{
	tn* root = (tn*)malloc(sizeof(tn));
	root->idx = node;
	root->child = 0;
	root->parent = NULL;
	
	buildTree(adj, V, root);
	
	return root;
}

tn* createTree(int** adj, int V, int choice)
{
	int i, j;
	q* leaves = (q*)malloc(sizeof(q));
	memset(leaves->arr, -1, sizeof(leaves->arr));
	leaves->front = leaves->rear = -1;
	
	int* isLeaf = (int*)malloc(V * sizeof(int));
	memset(isLeaf, 0, V * sizeof(int));
	
	int* in = (int*)malloc(V * sizeof(int));
	memset(in, 0, V * sizeof(int));
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(adj[i][j] == 1)
				in[j] += 1;
	
	for(i = 0; i < V; i++)
		if(in[i] < 2)
		{
			isLeaf[i] = 1;
			push(leaves, i);
		}
	
	j=0;
	while(j < V - 2)
	{
		int leaf = dequeue(leaves);
		j++;
		
		for(i = 0; i < V; i++)
			if(adj[leaf][i] == 1 and !isLeaf[i])
			{
				in[i] -= 1;
				if(in[i] < 2) 
					push(leaves, i);
			}
	
	}

	return (choice == 0 or leaves->arr[leaves->front + 1] == -1)? rootTree(adj, V, leaves->arr[leaves->front]): rootTree(adj, V, leaves->arr[leaves->front + 1]);
}

void eulerianTour(tn* node, int* tour ,int* idx, int* depth, int level, int* last)
{
	tour[*idx] = node->idx;
	depth[*idx] = level;
	
	*idx += 1;
	last[node->idx] = *idx - 1;
	
	int i;
	
	for(i = 0; i < node->child; i++)
	{
		eulerianTour(node->children[i], tour, idx, depth, level + 1, last);
		
		tour[*idx] = node->idx;
		depth[*idx] = level;
		
		*idx += 1;
		last[node->idx] = *idx - 1;
	}
}

int cmpfunc(const void* str1, const void* str2)
{
	const char** s1 = (const char**)str1;
	const char** s2 = (const char**)str2;
	
	return strlen(*s2) - strlen(*s1);
}

void findEncodingUtils(tn* node)
{
	if(node->child == 0)
		strcpy(node->str, "()");
	else
	{
		int i;
		char* substr[node->child];
		for(i = 0; i < node->child; i++)
		{
			findEncodingUtils(node->children[i]);
			substr[i] = node->children[i]->str;
		}
		qsort(substr, node->child, sizeof(char*), cmpfunc);
		
		for(i = 1; i < node->child; i++)
			strcat(substr[0], substr[i]);
			
		char tmp[N] = "(";
		strcat(tmp, substr[0]);
		strcat(tmp, ")");
		
		strcpy(node->str, tmp);
	}
}

char* findEncoding(int** adj, int V, int choice)
{
	tn* root = createTree(adj, V, choice);
	findEncodingUtils(root);
	return root->str;
}

int lca(int a, int b, int** adj, int V, int choice)
{
	int depth[N];
	int tour[N];
	int idx = 0;
	int last[N];
	
	tn* root = createTree(adj, V, choice);
	eulerianTour(root, tour, &idx, depth, 0, last);	
	
	int lower = (last[a] > last[b])? last[b]: last[a];
	int upper = (last[a] > last[b])? last[a]: last[b];	
	
	int res = rangeQueryOverlap(depth, idx, min, lower, upper, true);
	
	return tour[res];
}

int** revAdjacency(int** adj, int V)
{
	int i, j;
	int** radj;
	radj = (int**)malloc(sizeof(int*) * V);
	
	for(i = 0; i < V; i++)
		radj[i] = (int*)malloc(sizeof(int) * V);
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			radj[j][i] = adj[i][j];
	
	return radj;
}

void kosarajuUtility(bool* visited, int** adj, int V, int* who, int node)
{
	int i;
	visited[node] = true;
	
	for(i = 0; i < V; i++)
		if(adj[node][i] < INF and !visited[i])
		{
			who[i] = who[node];
			kosarajuUtility(visited, adj, V, who, i);
		}
		
}

int* sccKosaraju(int** adj, int V)
{
	int i;
	int* ordering;
	ordering = topSort(adj, V);
	
	int** radj;
	radj = revAdjacency(adj, V);
	
	bool* visited;
	visited = (bool*)malloc(sizeof(bool) * V);
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	int* who;
	who = (int*)malloc(sizeof(int) * V);
	memset(who, -1, sizeof(int) * V);
	
	for(i = 0; i < V; i++)
		if(!visited[ordering[i]])
		{
			who[ordering[i]] = ordering[i];
			kosarajuUtility(visited, radj, V, who, ordering[i]);
		}
		
	return who;
}

void tarjandfs(int** adj, bool* visited, int V, int node, int* ordering, int* counter)
{
	visited[node] = true;
	ordering[*counter] = node;
	*counter += 1;
	
	int i;
	
	for(i = 0; i < V; i++)
		if(!visited[i] and adj[node][i] < INF)
			tarjandfs(adj, visited, V, i, ordering, counter);
}

void tarjanUtility(S* stk, int** adj, int V, bool* visited, int* onStack, int* lowLink, int node, int* counter)
{
	int i, tmp;
	tmp = *counter;
	
	stk->arr[++stk->top] = node;
	
	visited[node] = true;
	onStack[node] = 1;

	lowLink[node] = tmp;
	*counter += 1;
	
	for(i = 0; i < V; i++)
		if(adj[node][i] < INF)
		{
			if(!visited[i])
				tarjanUtility(stk, adj, V, visited, onStack, lowLink, i, counter);
			if(onStack[i])
				lowLink[node] = min(lowLink[node], lowLink[i]);
		}
	
	if(lowLink[node] == tmp)
		while(stk->top != -1)
		{
			int pop = stk->arr[stk->top--];
			onStack[pop] = 0;
			
			if(pop == node)
				break;
			
			lowLink[pop] = tmp;	
		}

}

int* sccTarjan(int** adj, int V)
{
	int i, counter = 0;
	int* ordering;
	ordering = (int*)malloc(sizeof(int) * V);
	
	int* who;
	who = (int*)malloc(sizeof(int) * V);
	
	S* stk = (S*)malloc(sizeof(S));
	memset(stk->arr, 0, sizeof(stk->arr));
	stk->top = -1;
	
	bool* visited;
	visited = (bool*)malloc(sizeof(bool) * V);
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	int* onStack;
	onStack = (int*)malloc(sizeof(int) * V);
	
	int* lowLink;
	lowLink = (int*)malloc(sizeof(int) * V);
	
	for(i = 0; i < V; i++)
		if(!visited[i])
			tarjandfs(adj, visited, V, i, ordering, &counter);
	
	counter = 0;
	for(i = 0; i < V; i++)
		visited[i] = false;
	
	for(i = 0; i < V; i++)
		if(!visited[i])
			tarjanUtility(stk, adj, V, visited, onStack, lowLink, i, &counter);
	
	for(i = 0; i < V; i++)
		who[i] = ordering[lowLink[i]];
	
	return who;
}

S* find(int* unionFind, int a)
{
	S* stk = (S*)malloc(sizeof(S));
	stk->top = -1;
	
	stk->arr[++stk->top] = a;
	int tmpr = unionFind[a];
	
        if(unionFind[tmpr] == tmpr)
        {
                stk->arr[++stk->top] = tmpr;
                return stk;
        }
  
	while(unionFind[tmpr] != tmpr)
	{
		stk->arr[++stk->top] = tmpr;
		tmpr = unionFind[tmpr];
	}
	
	return stk;
}

void uni(int* unionFind, int a, int b, int* countCmp)
{
	S* ra = find(unionFind, a);
	S* rb = find(unionFind, b);
	
	if(ra->arr[ra->top] != rb->arr[rb->top])
	{
		if(ra->top <= rb->top)
			while(ra->top != -1)
			{
				int pop = ra->arr[ra->top--];
				unionFind[pop] = rb->arr[rb->top];
			}
		else
			while(rb->top != -1)
			{
				int pop = rb->arr[rb->top--];
				unionFind[pop] = ra->arr[ra->top];
			}
		*countCmp -= 1;
	}
}

int lbit(int x)
{
	return x & -x;
}

int calpfx(int* ftree, int x)
{
	int sum = 0;
	while(x != 0)
	{
		sum += ftree[x];
		x -= lbit(x);
	}
	
	return sum;
}

void pointUpdate(int* ftree, int S, int idx, int x)
{
	while(idx <= S)
	{
		ftree[idx] += x;
		idx += lbit(idx);
	}
}

int rangeCascading(int* ftree, int lower, int upper)
{
	return calpfx(ftree, upper + 1) - calpfx(ftree, lower);
}

int* createFTree(int* arr, int S)
{
	int i;
	int* ftree = (int*)malloc(sizeof(int) * (S + 1));
	
	for(i = 1; i <= S; i++)
		ftree[i] = arr[i - 1];
		
	for(i = 1; i < S; i++)
	{
		if(i + lbit(i) <= S)
			ftree[i + lbit(i)] += ftree[i];
	}
	
	return ftree;
}

int max(int a, int b)
{
	return (a > b)? a: b;
}

void findDepth(tn* node, int depth, int* maxDepth, int* parentUtils, int* depthCache)
{
	*maxDepth = max(*maxDepth, depth);
	int i;
	depthCache[node->idx] = depth;
	
	for(i = 0; i < node->child; i++)
	{
		parentUtils[node->children[i]->idx] = node->idx;
		findDepth(node->children[i], depth + 1, maxDepth, parentUtils, depthCache);
	}

}

struct t{
	int** pTable;
	int* dCache;
};
typedef struct t tuple;

tuple* createParentTable(tn* root, int V)
{
	int ** parent;
	int i, j;
	int depth = -1;
	int* parentUtils = (int*)malloc(sizeof(int) * V);
	int* depthCache = (int*)malloc(sizeof(int) * V);
	
	tuple* tup = (tuple*)malloc(sizeof(tuple));
	
	parentUtils[root->idx] = -1;
	findDepth(root, 0, &depth, parentUtils, depthCache);
	
	int logDepth = log(depth) / log(2);
	
	parent = (int**)malloc(sizeof(int*) * V);
	for(i = 0; i < V; i++)
		parent[i] = (int*)malloc(sizeof(int) * logDepth);
	
	for(i = 0; i < V; i++)
		parent[i][0] = parentUtils[i];
	
	for(i = 0; i < V; i++)
		for(j = 1; j < logDepth; j++)
			parent[i][j] = parent[parent[i][j - 1]][j - 1];
			
	tup->dCache = depthCache;
	tup->pTable = parent;

	return tup;
}

int binaryLifting(int a, int b, int** adj, int V, int choice)
{
	int ra, rb;
	tn* root = createTree(adj, V, choice);
	tuple* tup = createParentTable(root, V);
	
	int** parent = tup->pTable;
	int* depth = tup->dCache;
	ra = a;
	rb = b;
	
	if(depth[a] < depth[b])
	{
		int res = depth[b] - depth[a];
		int p = log(res) / log(2);
		int i;
		for(i = p; i >= 0; i--)
			if(res & (1 << i))
				rb = parent[rb][i];
	}
	
	if(depth[a] > depth[b])
	{
		int res = depth[a] - depth[b];
		int p = log(res) / log(2);
		int i;
		for(i = p; i >= 0; i--)
			if(res & (1 << i))
				ra = parent[ra][i];
	}
	
	int res = depth[ra];
	int p = log(res) / log(2);
	int i;
	
	for(i = p; i >= 0; i--)
		if(parent[ra][i] != parent[rb][i])
		{
			ra = parent[ra][i];
			rb = parent[rb][i];
		}
	
	return parent[ra][0];
}

void fillSegmentTree(int* tree, int idx, int S)
{
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;

	if(left < S and right < S)
	{
		fillSegmentTree(tree, left, S);
		fillSegmentTree(tree, right, S);
		tree[idx] = tree[left] + tree[right];
	}
}

int* createSegmentTree(int* arr, int S)
{
	int p = log(S) / log(2);
	if((1 << p) < S)
		p += 1;
	p = 1 << p;
	int* sTree = (int*)malloc(sizeof(int) * (2*p - 1));
	memset(sTree, 0, sizeof(int) * (2*p - 1));
	int i;
	
	for(i = 0; i < S; i++)
		sTree[p - 1 + i] = arr[i];
	
	fillSegmentTree(sTree, 0, (2*p - 1));
}

bool intersects(int a, int b, int c, int d)
{
	return (c >= a and c <= b) or (d >= a and d <= b) or (a >= c and a <= d) or (b >= c and b <= d);
}

int sumQuery(int* tree, int idx, int lower, int upper, int left, int right)
{
	if(lower >= left and upper <= right)
		return tree[idx];
	
	if(intersects(lower, upper, left, right))
		return sumQuery(tree, (2 * idx + 1), lower, lower + (upper - lower - 1) / 2, left, right) + sumQuery(tree, (2 * idx + 2), lower + (upper - lower - 1) / 2 + 1, upper, left, right);
	else
		return 0;
		
}

void updateQuery(int* tree, int idx, int x, int S)
{
	int p = log(S) / log(2);
	if((1 << p) < S)
		p += 1;
	p = 1 << p;
	
	int parent = p - 1 + idx;
	while(parent >= 0)
	{
		tree[parent] += x;
		if(parent == 0)
			break;
		parent = floor((parent - 1) / 2);
	}

}

int rangeQuerySegment(int* tree, int left, int right, int S)
{
	int p = log(S) / log(2);
	if((1 << p) < S)
		p += 1;
	p = 1 << p;
	
	return sumQuery(tree, 0, (p - 1), (2 * p - 1), (p - 1 + left), (p - 1 + right));
}

bool fordFulklersonDFS(int* path, int* idx, bool* visited, int** rGraph, int V, int node, int target, int* maxFlow, int delta)
{
	int i;
	visited[node] = true;
	path[*idx] = node;
	*idx += 1;
	
	if(node == target)
	{
		int bottleneck = INT_MAX;
		for(i = 0; i < *idx - 1; i++)
			bottleneck = min(bottleneck, rGraph[path[i]][path[i + 1]]);
		
		*maxFlow += bottleneck;
		for(i = 0; i < *idx - 1; i++)
		{
			rGraph[path[i]][path[i + 1]] -= bottleneck;
			rGraph[path[i + 1]][path[i]] += bottleneck;
		}
		
		return true;
	}
	else
	{
		for(i = 0; i < V; i++)
			if(!visited[i] and rGraph[node][i] >= delta)
				if(!fordFulklersonDFS(path, idx, visited, rGraph, V, i, target, maxFlow, delta))
				{
					visited[i] = false;
					*idx -= 1;
				}
				else
					return true;
				
		return false;
	}
}

int findDelta(int** adj, int V)
{
	int i,j;
	int delta = INT_MIN;
	
	for(i=0;i<V;i++)
		for(j=0;j<V;j++)
			delta = max(delta, adj[i][j]);
	
	int pow = log(delta) / log(2);
	return 1 << pow;
}

int fordFulkerson(int** adj, int V, int start, int target)
{
	int* path = (int*)malloc(sizeof(int) * V);
	int idx = 0;
	
	bool* visited = (bool*)malloc(sizeof(bool) * V);
	memset(visited, false, sizeof(bool) * V);
	
	int maxFlow = 0;
	int delta = findDelta(adj, V);
	
	while(delta != 0)
	{
		while(fordFulklersonDFS(path, &idx, visited, adj, V, start, target, &maxFlow, delta))
		{
			idx = 0;
			memset(visited, false, sizeof(bool) * V);
		}
		delta /= 2;
	}
	return maxFlow;
}

bool edmondsKarpBFS(int** rGraph, int V, int start, int target, int* maxFlow, int delta)
{
	bool* visited = (bool*)malloc(sizeof(bool) * V);
	memset(visited, false, sizeof(bool) * V);
	
	int* parent = (int*)malloc(sizeof(int) * V);
	
	int i;
	q* que = (q*)malloc(sizeof(q));
	que->front = que->rear = -1;
	
	push(que, start);
	visited[start] = true;
	parent[start] = -1;
	
	while(!isEmpty(que))
	{
		int node = dequeue(que);
		for(i = 0; i < V; i++)
			if(rGraph[node][i] >= delta and !visited[i])
			{
				push(que, i);
				parent[i] = node;
				visited[i] = true;
			}
		if(visited[target])
			break;
	}
	
	if(!visited[target])
		return false;
	
	int bottleneck = INT_MAX;
	int node = target;
	
	while(parent[node] != -1)
	{
		bottleneck = min(bottleneck, rGraph[parent[node]][node]);
		node = parent[node];
	}
	
	node = target;
	while(parent[node] != -1)
	{
		rGraph[parent[node]][node] -= bottleneck;
		rGraph[node][parent[node]] += bottleneck;
		
		node = parent[node];
	}
	
	*maxFlow += bottleneck;
	return true;
}

int edmondsKarp(int** adj, int V, int start, int target)
{
	int maxFlow = 0;
	int delta = findDelta(adj, V);
	
	while(delta != 0)
	{
		while(edmondsKarpBFS(adj, V, start, target, &maxFlow, delta)){
		}

		delta /= 2;
	}
	return maxFlow;
}

bool dinicDFS(int* path, int* idx, bool* visited, int** rGraph, int** lGraph, int V, int node, int target, int* maxFlow)
{
	int i;
	visited[node] = true;
	path[*idx] = node;
	*idx += 1;
	
	if(node == target)
	{
		int bottleneck = INT_MAX;
		for(i = 0; i < *idx - 1; i++)
			bottleneck = min(bottleneck, rGraph[path[i]][path[i + 1]]);
		
		*maxFlow += bottleneck;
		for(i = 0; i < *idx - 1; i++)
		{
			rGraph[path[i]][path[i + 1]] -= bottleneck;
			rGraph[path[i + 1]][path[i]] += bottleneck;
			
			lGraph[path[i]][path[i + 1]] -= bottleneck;
			lGraph[path[i + 1]][path[i]] += bottleneck;
		}
		
		return true;
	}
	else
	{
		for(i = 0; i < V; i++)
			if(!visited[i] and rGraph[node][i])
				if(!dinicDFS(path, idx, visited, rGraph, lGraph, V, i, target, maxFlow))
				{
					visited[i] = false;
					*idx -= 1;
				}
				else
					return true;
				
		return false;
	}
}

bool dinicBFS(int** rGraph, int** lGraph, int V, int start, int target, int delta)
{
	bool* visited = (bool*)malloc(sizeof(bool) * V);
	memset(visited, false, sizeof(bool) * V);
	
	int i;
	q* que = (q*)malloc(sizeof(q));
	que->front = que->rear = -1;
	
	push(que, start);
	visited[start] = true;
	
	while(!isEmpty(que))
	{
		int node = dequeue(que);
		for(i = 0; i < V; i++)
			if(rGraph[node][i] >= delta and !visited[i])
			{
				push(que, i);
				lGraph[node][i] = rGraph[node][i];
				visited[i] = true;
			}
		if(visited[target])
			break;
	}
	
	return visited[target];
}

int dinic(int** adj, int V, int start, int target)
{
	int i, maxFlow = 0;
	int delta = findDelta(adj, V);
	
	int* path = (int*)malloc(sizeof(int) * V);
	int idx = 0;
	
	bool* visited = (bool*)malloc(sizeof(bool) * V);
	memset(visited, false, sizeof(bool) * V);
	
	int** lGraph = (int**)malloc(sizeof(int*) * V);
	for(i = 0; i < V; i++)
	{
		lGraph[i] = (int*)malloc(sizeof(int) * V);
		memset(lGraph[i], 0, sizeof(int) * V);
	}
	
	while(delta > 0)
	{
		while(dinicBFS(adj, lGraph, V, start, target, delta))
		{
			while(dinicDFS(path, &idx, visited, lGraph, adj, V, start, target, &maxFlow))
			{
				idx = 0;
				memset(visited, false, sizeof(bool) * V);
			}
			
			for(i = 0; i < V; i++)
					memset(lGraph[i], 0, sizeof(int) * V);
		}
		delta /= 2;
	}
	return maxFlow;
}

bool findNext(int* excessFlow, int* height, int V, int* node, int start, int target)
{
	int count = 0;
	int i, next = INT_MIN;
	
	for(i = 0; i < V; i++)
		if(excessFlow[i] > 0 and i != start and i != target)
		{
			count++;
			if(next < height[i])
			{
				next = height[i];
				*node = i;
			}
		}

	return count > 0;
}

int pushRelabel(int** adj, int V, int start, int target)
{
	int i, node;
	int* height = (int*)malloc(sizeof(int) * V);
	memset(height, 0, sizeof(int) * V);
	height[start] = V;
	
	int* excessFlow = (int*)malloc(sizeof(int) * V);
	memset(excessFlow, 0, sizeof(int) * V);
	for(i = 0; i < V; i++)
		if(adj[start][i])
		{
			adj[i][start] = adj[start][i];
			excessFlow[i] = adj[start][i];
			adj[start][i] = 0;
		}
	
	while(findNext(excessFlow, height, V, &node, start, target))
	{
		bool flag = false;
		for(i = 0; i < V; i++)
		{
			if(height[i] == height[node] - 1 and adj[node][i])
			{
				flag = true;
				int flowPushed = min(adj[node][i], excessFlow[node]);
				
				adj[node][i] -= flowPushed;
				adj[i][node] += flowPushed;
				
				excessFlow[node] -= flowPushed;
				excessFlow[i] += flowPushed;
				
				break;
			}
		}
		if(!flag)
		{
			int adjMin = INT_MAX;
			
			for(i = 0; i < V; i++)
				if(adj[node][i] and adjMin > height[i])
					adjMin = height[i];
				
			height[node] = adjMin + 1;
		}
	}
	
	return excessFlow[target];
}

void augmentMatching(int* matched_A, int* matched_B, int* path, int pathLen, int partition)
{
	int i;
	for(i = 0; i < pathLen; i += 2)
	{
		matched_B[path[i] - partition] = path[i + 1];
		matched_A[path[i + 1]] = path[i];
	}
}

bool createLevelGraph(q** lGraph, int** adj, int* matched_A, int* matched_B, int partition, int V, int* qlen)
{
	int i, j;
	bool* visited = (bool*)malloc(sizeof(bool) * (V - partition));
	memset(visited, false, sizeof(bool) * (V - partition));
	
	memset(lGraph, (int)none, sizeof(q*) * V);
	lGraph[0] = (q*)malloc(sizeof(q));
	lGraph[0]->front = -1;
	lGraph[0]->rear = -1;
	*qlen += 1;
	
	for(i = 0; i < partition; i++)
		if(matched_A[i] == -1)
			push(lGraph[0], i);
	
	i = 0;
	bool flag = true;
	
	while(lGraph[i] != none and flag)
	{
		int size = lGraph[i]->rear + 1;
		bool set = false;

		while(size > 0)
		{
			int node = dequeue(lGraph[i]);
			if(i % 2)
			{
				if(!set)
				{
					lGraph[i + 1] = (q*)malloc(sizeof(q));
					lGraph[i + 1]->front = -1;
					lGraph[i + 1]->rear = -1;
					*qlen += 1;
					set = true;
				}
				push(lGraph[i + 1], matched_B[node - partition]);
			}
			else
			{
				for(j = 0; j < (V - partition); j++)
					if(adj[node][partition + j] and matched_B[j] != node and !visited[j])
					{
						if(!set)
						{
							lGraph[i + 1] = (q*)malloc(sizeof(q));
							lGraph[i + 1]->front = -1;
							lGraph[i + 1]->rear = -1;
							*qlen += 1;
							set = true;
						}
						push(lGraph[i + 1], partition + j);
						visited[j] = true;
						
						if(matched_B[j] == -1)
							flag = false;
					}
					
			}
			size--;
		}
		i++;
	}
	
	return !flag;
}

bool findPath(bool* visited, q** lGraph, int* matched_A, int* matched_B, int qlen, int V, int partition, int** adj, int* path, int* path_idx)
{
	S* stk = (S*)malloc(sizeof(S));
	stk->top = -1;

	int i, j;
	for(i = 0; i <= lGraph[qlen - 1]->rear; i++)
	{
		int node = lGraph[qlen - 1]->arr[i];	
		if(!visited[node] and matched_B[node - partition] == -1)
		{
			stk->arr[++stk->top] = node;
			break;
		}
	}
	
	if(stk->top == -1)
		return false;
	
	i = qlen - 1;
	while(stk->top != -1)
	{
		int node = stk->arr[stk->top--];
		
		visited[node] = true;
		path[*path_idx] = node;
		*path_idx += 1;	
		
		if(i == 0)
			break;	
		
		for(j = 0; j <= lGraph[i - 1]->rear; j++)
		{
			int prev = lGraph[i - 1]->arr[j];
			if((qlen - i)%2)
			{
				if(adj[node][prev] and matched_A[prev] != node and !visited[prev])
				{
					stk->arr[++stk->top] = prev;
					i--;
					break;
				}
			}
			else
			{
				if(!visited[matched_A[node]])
				{
					stk->arr[++stk->top] = matched_A[node];
					i--;
				}
				break;
			}
		}
	}

	return i == 0;
}

int* hopcroftKarp(int** adj, int V, int partition)
{
	int* matched_A = (int*)malloc(sizeof(int) * partition);
	memset(matched_A, -1, sizeof(int) * partition);
	
	int* matched_B = (int*)malloc(sizeof(int) * (V - partition));
	memset(matched_B, -1, sizeof(int) * (V - partition));
	
	q** lGraph = (q**)malloc(sizeof(q*) * V);
	int qlen = 0;

	while(createLevelGraph(lGraph, adj, matched_A, matched_B, partition, V, &qlen))
	{
		bool* visited = (bool*)malloc(sizeof(bool) * V);
		memset(visited, false, sizeof(bool) * V);
		
		int* path = (int*)malloc(sizeof(int) * V);
		int path_idx = 0;
		
		while(findPath(visited, lGraph, matched_A, matched_B, qlen, V, partition, adj, path, &path_idx))
		{
			augmentMatching(matched_A, matched_B, path, path_idx, partition);
			path_idx = 0;
		}
		qlen = 0;
	}
	
	return matched_A;
}

struct degree_node{
	int degree;
	int idx;
};
typedef struct degree_node didx;

int comparator(const void* a, const void* b)
{
	didx** x = (didx**)a;
	didx** y = (didx**)b;
	
	return (*y)->degree - (*x)->degree;
}

int welshPowell(int** adj, int V)
{
	int i, j, k, colors = 0;
	didx** nodes = (didx**)malloc(sizeof(didx*) * V);
	
	for(i = 0; i < V; i++)
	{
		int degree_count = 0;
		for(j = 0; j < V; j++)
			if(adj[i][j])
				degree_count++;
		
		nodes[i] = (didx*)malloc(sizeof(didx));		
		nodes[i]->degree = degree_count;
		nodes[i]->idx = i;
	}
	
	qsort(nodes, V, sizeof(didx*), comparator);

	int* colored = (int*)malloc(sizeof(int) * V);
	memset(colored, -1, sizeof(int) * V);
	
	q* cache = (q*)malloc(sizeof(q));
	cache->front = -1;
	cache->rear = -1;
	
	for(i = 0; i < V; i++)
	{
		int node = nodes[i]->idx;
		if(colored[node] == -1)
		{
			colored[node] = colors;
			cache->front = -1;
			cache->rear = -1;
			push(cache, node);
			
			for(j = i + 1; j < V; j++)
			{
				int node_to_be_colored = nodes[j]->idx;
				if(colored[node_to_be_colored] == -1)
				{
					int size = cache->rear + 1;
					bool canColor = true;
					
					for(k = 0; k < size; k++)
					{
						if(adj[node_to_be_colored][cache->arr[k]])
						{
							canColor = false;
							break;
						}
					}
					
					if(canColor)
					{
						colored[node_to_be_colored] = colors;
						push(cache, node_to_be_colored);
					}
				}
			}
			colors ++;
		}
	}
	return colors;
}

struct trie_node{
	struct trie_node* chr[26];
	
	bool is_end;
	char* word_formed;
	
	struct trie_node* failure_link;
	struct trie_node* dictionary_link;
};
typedef struct trie_node trie;

trie* getTrieNode()
{
	trie* tnode = (trie*)malloc(sizeof(trie));
	tnode->dictionary_link = none;
	
	tnode->failure_link = none;
	tnode->is_end = false;
	
	tnode->word_formed = "";
	memset(tnode->chr, (int)none, sizeof(tnode->chr));
	return tnode;
}

void setFailureLink(trie* tnode, trie* fl)
{
	tnode->failure_link = fl;
}

void setDictionaryLink(trie* tnode, trie* dl)
{
	tnode->dictionary_link = dl;
}

void setEnd(trie* tnode)
{
	tnode->is_end = true;
}

void setWord(trie* tnode, char* str)
{
	tnode->word_formed = str;
}

bool hasLink(trie* tnode, int pos)
{
	return tnode->chr[pos] != none;
}

char* substr(char* src, int i, int j)
{
	int counter = 0;
	char* dest = (char*)malloc(sizeof(char) * (j - i + 1));
	while(i < j and i <= strlen(src) - 1)
	{
		dest[counter++] = *(src + i);
		i++;	
	}
	
	dest[counter] = '\0';
	return dest;
}

void addVocab(trie* tnode, char* vocab)
{
	int i;
	trie* node = tnode;
	
	for(i = 0; i < strlen(vocab); i++)
	{
		if(!hasLink(node, vocab[i] - 'a'))
		{
			trie* new_node = getTrieNode();
			setWord(new_node, substr(vocab, 0, i + 1));
			
			node->chr[vocab[i] - 'a'] = new_node;
			node = new_node;
		}
		else
			node = node->chr[vocab[i] - 'a'];
	}
	setEnd(node);
}

trie* getWord(trie* root, char* word)
{
	int i;
	trie* node = root;
	
	for(i = 0; i < strlen(word); i++)
	{
		if(hasLink(node, word[i] - 'a'))
			node = node->chr[word[i] - 'a'];
		else
			return none;
	}
	
	return node;
}

bool hasWord(trie* root, char* word)
{
	trie* node = getWord(root, word);
	return !(node == none) and node->is_end;
}

void addFailureLinks(trie* node, trie* root)
{
	int i;
	if(node == root)
		setFailureLink(node, root);
		
	for(i = 1; i <= strlen(node->word_formed); i++)
	{
		trie* fl = getWord(root, substr(node->word_formed, i, strlen(node->word_formed)));
		if(!(fl == none))
		{
			setFailureLink(node, fl);
			break;
		}
	}
	
	for(i = 0; i < 26; i++)
		if(hasLink(node, i))
			addFailureLinks(node->chr[i], root);
}

void addDictionaryLinks(trie* node, trie* root)
{
	trie* temp = node;
	int i;
	
	while(temp->failure_link != root)
	{
		temp = temp->failure_link;
		if(temp->is_end)
		{
			setDictionaryLink(node, temp);
			break;
		}
	}
	
	for(i = 0; i < 26; i++)
		if(hasLink(node, i))
			addDictionaryLinks(node->chr[i], root);
}

trie* create_trie_aho_corasick(char** vocabulary, int vocab_size)
{
	trie* root = getTrieNode();
	int i;
	
	for(i = 0; i < vocab_size; i++)
		addVocab(root, vocabulary[i]);
		
	addFailureLinks(root, root);
	addDictionaryLinks(root, root);
	
	return root;
}

int getHashCode(char* str, int hashLength)
{
	int i;
	int hash_code = 0;
	
	for(i = 0; i < strlen(str); i++)
		hash_code += 1 << (str[i] - 'a');
		
	return hash_code % hashLength;
}

int* search_for_words(trie* root, char* search_word, char** vocabulary, int V)
{
	int i;
	const int HASH_LENGTH = 2098;
	
	int* res = (int*)malloc(sizeof(int) * HASH_LENGTH);
	memset(res, 0, sizeof(int) * HASH_LENGTH);
	
	for(i = 0; i < strlen(search_word); i++)
	{
		if(hasLink(root, search_word[i] - 'a'))
			root = root->chr[search_word[i] - 'a'];
		else
		{
			while(root != root->failure_link and !hasLink(root, search_word[i] - 'a'))
				root = root->failure_link;
			
			if(hasLink(root, search_word[i] - 'a'))
				root = root->chr[search_word[i] - 'a'];
			else
				continue;
		}
		
		if(root->is_end)
			res[getHashCode(root->word_formed, HASH_LENGTH)]++;
		
		trie* dict_link = root->dictionary_link;
		if(!(dict_link == none))
			res[getHashCode(dict_link->word_formed, HASH_LENGTH)]++;
			dict_link = dict_link->dictionary_link;
	}
	
//	for(i = 0; i < V; i++)
//		printf("%s - %d\n", vocabulary[i], res[getHashCode(vocabulary[i], HASH_LENGTH)]);

	return res;
}

bool* knuth_morris_pratt(char* str, char* pattern)
{
	int* pre_suff = (int*)malloc(sizeof(int) * strlen(pattern));
	memset(pre_suff, 0, sizeof(int) * strlen(pattern));
	
	int i = 0, j = 1;
	while(j < strlen(pattern))
	{
		if(pattern[i] == pattern[j])
		{
			pre_suff[j] = i + 1;
			i++;
			j++;
		}
		else
		{
			while(i != 0 and pattern[i] != pattern[j])
				i = pre_suff[i - 1];
			if(i == 0 and pattern[i] != pattern[j])
				pre_suff[j] = 0;
			else
			{
				pre_suff[j] = i + 1;
				i++;
			}
			j++;
		}
	}
	
	bool* occurance = (bool*)malloc(sizeof(bool) * strlen(str));
	memset(occurance, false, sizeof(bool) * strlen(str));
	
	i = 0, j = 0;
	while(i < strlen(str))
	{
		while(j < strlen(pattern) and str[i] == pattern[j])
		{
			i++;
			j++;
		}
		
		if(j == strlen(pattern))
		{
			occurance[i - j] = true;
			j = pre_suff[j - 1];
		}
		
		else if(i < strlen(str) and str[i] != pattern[j])
		{
			if(j != 0)
				j = pre_suff[j - 1];
			else
				i++;
		}
	}

	return occurance;
}

int* calculate_z_value(char* str)
{
	int l = 1, r;
	int i = 1;
	
	int* z_values = (int*)malloc(sizeof(int) * strlen(str));
	z_values[0] = strlen(str);
	while(str[i] == str[0])
		i++;
	z_values[1] = i - 1;
	r = i - 1;
	
	for(i = 2; i < strlen(str); i++)
	{
		if(i > r)
		{
			int j = i, k = 0;
			while(str[k] == str[j])
			{
				j++;
				k++;
			}
			z_values[i] = k;
			if(k > r)
			{
				r = k;
				l = i;
			}
		}
		else
		{
			int k = i - l;
			int beta = r - i + 1;
			if(z_values[k] < beta)
				z_values[i] = z_values[k];
			else if(z_values[k] > beta)
				z_values[i] = beta;
			else
			{
				z_values[i] = beta;
				k = r + 1;
				while(str[beta] == str[k])
				{
					beta++;
					k++;
					z_values[i]++;
				}
                                l = i;
                                r = k - 1;
			}	
		}
	}
	
	return z_values;
}

bool* z_algorithm(char* str ,char* pattern)
{
	const char* seperator = "$";
	int i, pattern_length = strlen(pattern);
	
	strcat(pattern, seperator);
	strcat(pattern, str);
	
	int* z_values = calculate_z_value(pattern);
	bool* occurance = (bool*)malloc(sizeof(bool) * strlen(str));
	memset(occurance, false, sizeof(bool) * strlen(str));
	
	for(i = 0; i < strlen(str); i++)
		if(z_values[pattern_length + i + 1] == pattern_length)
			occurance[i] = true;
		
	return occurance;
}

int** transform_matrix(int** matrix, int V)
{
	int i, j;
	int** new_matrix = (int**)malloc(sizeof(int*) * V);
	for(i = 0; i < V; i++)
		new_matrix[i] = (int*)malloc(sizeof(int) * V);
		
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			new_matrix[i][j] = matrix[i][j];
			
	int min_num;
	for(i = 0; i < V; i++)
	{
		min_num = INF;
		for(j = 0; j < V; j++)
		{
			min_num = min(min_num, new_matrix[i][j]);
			if(min_num == 0)
				break;
		}
		
		if(min_num != 0)
			for(j = 0; j < V; j++)
				new_matrix[i][j] -= min_num;
	}
	
	for(i = 0; i < V; i++)
	{
		min_num = INF;
		for(j = 0; j < V; j++)
		{
			min_num = min(min_num, new_matrix[j][i]);
			if(min_num == 0)
				break;
		}
		
		if(min_num != 0)
			for(j = 0; j < V; j++)
				new_matrix[j][i] -= min_num;
	}	
	
	return new_matrix;	
}

int calculate_lines(int** line_matrix, int** matrix, int V)
{
	int number_of_lines = 0;
	int i, j, number_of_zeros = 0;
	
	int* rows = (int*)malloc(sizeof(int) * V);
	memset(rows, 0, sizeof(int) * V);
	
	int* cols = (int*)malloc(sizeof(int) * V);
	memset(cols, 0, sizeof(int) * V);
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(matrix[i][j] == 0)
			{
				rows[i]++;
				cols[j]++;
				number_of_zeros++;
			}
	
	while(number_of_zeros > 0)
	{
		int max_row_idx, max_col_idx;
		int max_row = -INF, max_col = -INF;
		
		for(i = 0; i < V; i++)
			if(max_row < rows[i])
			{
				max_row = rows[i];
				max_row_idx = i;
			}
		
		for(i = 0; i < V; i++)
			if(max_col < cols[i])
			{
				max_col = cols[i];
				max_col_idx = i;
			}
		
		if(max_row > max_col)
		{
			number_of_zeros -= rows[max_row_idx];
			rows[max_row_idx] = 0;
			for(i = 0; i < V; i++)
			{
				line_matrix[max_row_idx][i] += 1;
				if(matrix[max_row_idx][i] == 0)
					cols[i] -= 1;
			}
		}
		else
		{
			number_of_zeros -= cols[max_col_idx];
			cols[max_col_idx] = 0;
			for(i = 0; i < V; i++)
			{
				line_matrix[i][max_col_idx] += 1;
				if(matrix[i][max_col_idx] == 0)
					rows[i] -= 1;
			}
		}
		number_of_lines += 1;
	}
	return number_of_lines; 
}

int hungarian(int** adj, int V)
{
	int i, j;
	int res = 0;
	
	int** transformed_matrix = transform_matrix(adj, V);
	int** line_matrix = (int**)malloc(sizeof(int*) * V);
	for(i = 0; i < V; i++)
		line_matrix[i] = (int*)malloc(sizeof(int) * V);
	fillArray(line_matrix, V, V, 0);
	
	while(calculate_lines(line_matrix, transformed_matrix, V) != V)
	{
		int min_number = INF;
		for(i = 0; i < V; i++)
			for(j = 0; j < V; j++)
				if(!line_matrix[i][j] and min_number > transformed_matrix[i][j])
					min_number = transformed_matrix[i][j];
		for(i = 0; i < V; i++)
			for(j = 0; j < V; j++)
			{
				if(!line_matrix[i][j])
					transformed_matrix[i][j] -= min_number;
				if(line_matrix[i][j] == 2)
					transformed_matrix[i][j] += min_number;
			}
			
		fillArray(line_matrix, V, V, 0);
	}
	
	int** bGraph = (int**)malloc(sizeof(int*) * (2 * V));
	for(i = 0; i < 2 * V; i++)
		bGraph[i] = (int*)malloc(sizeof(int) * (2 * V));
	fillArray(bGraph, 2 * V, 2 * V, 0);
	
	for(i = 0; i < V; i++)
		for(j = 0; j < V; j++)
			if(!transformed_matrix[i][j])
			{
				bGraph[i][V + j] = 1;
				bGraph[V + j][i] = 1;
			}

	int* allocations = hopcroftKarp(bGraph, 2 * V, V);
	for(i = 0; i < V; i++)
		res += adj[i][allocations[i] - V];
	return res;
}

int main()
{
	//main()!!
}

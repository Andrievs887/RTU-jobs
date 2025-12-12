from graphviz import Digraph

# ---------- Node and BST classes ----------

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    # Insert a key into the BST
    def insert(self, key):
        def _insert(node, key):
            if node is None:
                return Node(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            # if key == node.key, do nothing (no duplicates)
            return node

        self.root = _insert(self.root, key)

    # Search for a key in the BST
    def search(self, key):
        def _search(node, key):
            if node is None:
                return None
            if key == node.key:
                return node
            if key < node.key:
                return _search(node.left, key)
            else:
                return _search(node.right, key)

        return _search(self.root, key)

    # Helper: find minimum value node in subtree
    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    # Delete a key from the BST
    def delete(self, key):
        def _delete(node, key):
            if node is None:
                return None

            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                # Node with only one child or no child
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left

                # Node with two children: get inorder successor
                succ = self._min_value_node(node.right)
                node.key = succ.key
                node.right = _delete(node.right, succ.key)

            return node

        self.root = _delete(self.root, key)

    # Inorder traversal (Left, Root, Right)
    def inorder(self):
        res = []

        def _in(node):
            if node:
                _in(node.left)
                res.append(node.key)
                _in(node.right)

        _in(self.root)
        return res

    # Preorder traversal (Root, Left, Right)
    def preorder(self):
        res = []

        def _pre(node):
            if node:
                res.append(node.key)
                _pre(node.left)
                _pre(node.right)

        _pre(self.root)
        return res

    # Postorder traversal (Left, Right, Root)
    def postorder(self):
        res = []

        def _post(node):
            if node:
                _post(node.left)
                _post(node.right)
                res.append(node.key)

        _post(self.root)
        return res


# ---------- Graphviz plotting ----------

def plot_bst(tree, filename="bst"):
    """Render the BST to a PNG file using Graphviz."""
    dot = Digraph()

    def _add(node):
        if not node:
            return
        dot.node(str(node.key), str(node.key))
        if node.left:
            dot.edge(str(node.key), str(node.left.key))
            _add(node.left)
        if node.right:
            dot.edge(str(node.key), str(node.right.key))
            _add(node.right)

    _add(tree.root)
    dot.render(filename, format="png", cleanup=True)


# ---------- Helper to build and show steps ----------

def build_and_show(sequence, prefix):
    bst = BST()
    print(f"\nBuilding tree for sequence {prefix}: {sequence}")
    for i, val in enumerate(sequence, start=1):
        bst.insert(val)
        print(f"After inserting {val}:")
        print("  Inorder  :", bst.inorder())
        print("  Preorder :", bst.preorder())
        print("  Postorder:", bst.postorder())
        plot_bst(bst, f"{prefix}_step_{i}")
    return bst


# ---------- Main logic for Task 1 ----------

if __name__ == "__main__":
    a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
    b = [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]
    c = [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]

    # Build trees and generate step-by-step plots
    bst_a = build_and_show(a, "tree_a")
    bst_b = build_and_show(b, "tree_b")
    bst_c = build_and_show(c, "tree_c")

    # Demonstrate search and delete on tree_a
    print("\n--- Search examples on tree_a ---")
    for key in [49, 60, 100]:
        node = bst_a.search(key)
        print(f"Search {key} ->", "found" if node else "not found")

    print("\n--- Delete examples on tree_a ---")
    for key in [13, 49]:
        print(f"\nBefore deleting {key}:")
        print("  Inorder:", bst_a.inorder())
        plot_bst(bst_a, f"tree_a_before_delete_{key}")

        bst_a.delete(key)

        print(f"After deleting {key}:")
        print("  Inorder:", bst_a.inorder())
        plot_bst(bst_a, f"tree_a_after_delete_{key}")

"""Provides the `Tree` class for creating trees of objects."""
from builtins import super, object
from collections import MutableSequence, Iterable
from overrides import overrides


class BadParentError(Exception):
    """Raised when a parent does not have a reference to a requested child."""


#TODO: Let trees have multiple parents?
class Tree(Iterable):
    """A tree with support for name fetching.
    
    A tree can be constructed by simply assigning `Tree`s to each other's
    attributes:
        
        >>> p = Tree()
        >>> # doctest does not run these tests in __main__, so we give p
        >>> # a sensible fallback name
        >>> p.fallback_name = 'root'
        >>> p.child_a = Tree()
        >>> p.child_b = Tree()
        >>> p.child_b.leaf = Tree()

    The `long_name()` method may then be used to get the full path of an 
    object within the tree.

        >>> print(p.child_a.long_name)
        root.child_a
        >>> print(p.child_b.leaf.long_name)
        root.child_b.leaf
    
    Objects in the tree must be unique. No object may be its own ancestor
    or sibling. Attempts to add a `Tree` attribute that would result in 
    such a graph instead result in that `Tree` being shallow-copied.

        >>> p = Tree()
        >>> p.attribute = object()
        >>> p.child = p
        >>> p.child is p
        False
        >>> p.child.attribute is p.attribute  # true for non-Tree attributes
        True

        >>> q = Tree()
        >>> q.num = 1
        >>> p.child_a = q
        >>> p.child_a is q
        True
        >>> p.child_b = q
        >>> p.child_b is q
        False
        >>> p.child_b.num
        1

    Attributes:
        fallback_name (str): The name that this object has when it is the
            highest object in the tree and does not have a unique global
            alias. Defaults to "unnamed".
    
    """

    def __init__(self):
        """Creates a new parentable with no parent."""
        super().__init__()
        self.fallback_name = "unnamed"
        self.__parent = None

    @property
    def parent(self):
        """Returns the direct parent.

        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> p.child = q
            >>> q.parent is p
            True

        """
        return self.__parent

    @parent.setter
    def parent(self, value):
        """Sets the parent of a `Tree`.

        If we already had a parent, unlinks `self` from its previous parent
        before linking it to the new one.

        NB: It is _highly unlikely_ that you will need to set this property
        yourself. Setting this property, unless overridden in some parent 
        class, does not register `self` as a child of `value`, which could
        cause unexpected behaviour.

        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> r = Tree()

            Changing the parent of `q` unlinks it from its previous parent:
            >>> p.child = q
            >>> q.parent = r
            >>> p.child
            Traceback (most recent call last):
                ...
            AttributeError: message
            >>> q in p.children
            False

            It does not make it appear in `r`'s list of children:
            >>> q in r.children
            False

        """
        if self.parent is not None:
            self.parent._unregister_child(self)
        self._on_new_parent(value)

    def _on_new_parent(self, new_parent):
        """Called when an object's direct parent is set.
        
        Sets `self.parent` to `new_parent`.
        
        """
        self.__parent = new_parent

    def _unregister_child(self, child):
        """Removes any links between `self` and `child`.
        
        Raises:
            BadParentError: if `child` is not a child of this `Tree`.

        """
        matches = tuple(k for k, v in self.__dict__.items() if v is child)
        if len(matches) == 0:
            raise BadParentError('{value} is not a child of {self}'
                    .format(value=child, self=self))
        assert len(matches) == 1  # no duplicate children!
        delattr(self, matches[0])

    @property
    def children(self):
        """Returns the children of this object at the time of calling.

        Returns:
            (tuple) The current children of the object.

        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> r = Tree()
            >>> p.child = q
            >>> p.other_child = r
            >>> len(p.children)
            2
            >>> q in p.children and r in p.children
            True

        """
        dict_ = dict(self.__dict__)
        if '_Tree__parent' in dict_:
            del dict_['_Tree__parent']
        return tuple(filter(lambda x: isinstance(x, Tree), dict_.values()))

    @property
    def highest_parent(self):
        """Returns the highest parent in the tree.

        Examples:
            An object at the top of the tree is its own highest parent.
            >>> p = Tree()
            >>> p.highest_parent is p
            True

            Objects lower down in the tree recurse up the tree to find their
            highest parent.
            >>> p.q = Tree()
            >>> p.q.r = Tree()
            >>> p.q.r.highest_parent is p
            True

            see that function's
            documentation for how argumewnt
        """
        if self.parent is None:
            return self
        else:
            return self.parent.highest_parent

    @property
    def long_name(self):
        """Gets the long name of `self`.

        The long name of an object is the concatenation of the names of its
        parents, starting from the highest parent.

        Returns:
            (str): The full path to the object in the tree.
        
        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> p.child = Tree()
            >>> p.child.leaf = q
            >>> print(q.long_name)  # we are not in __main__
            unnamed.child.leaf
        
        """
        if self.parent is None:
            return self.name
        else:
            return self.parent.long_name + self.name

    @property
    def name(self):
        """Gets the name by which this object's parent refers to it.
        
        If there is no parent, searches `globals()` for a unique alias. If
        a unique global alias cannot be found, returns `self.fallback_name`.
        
        Examples:
            Objects at the top of the tree return their name in the global
            namespace, if only one such name exists.
            >>> p = Tree()
            >>> # doctest does not run these tests in __main__, but if it
            >>> # did, this would return true.
            >>> not __name__ == '__main__' or str(p.name) == 'p'
            True

            If no unique global aliases to the root exists, `.fallback_name`
            is returned.
            >>> q = p  # now p has two global aliases, 'p' and 'q'
            >>> p.name == p.fallback_name
            True

            Otherwise, an object's name is the name its parent gives it,
            as return by `child.parent.name_of(child)`.
            >>> q = Tree()
            >>> p.child = q
            >>> print(q.name)
            .child
            
        """
        if self.parent is None:
            # did someone say namespace hacking?
            globals_ = {}
            if __name__ == '__main__':
                globals_ = globals()
            else:
                import __main__
                globals_ = __main__.__dict__
            aliases = tuple(filter(lambda x: x[1] is self, globals_.items()))
            return aliases[0][0] if len(aliases) == 1 else self.fallback_name
        else:
            return self.parent.name_of(self)

    def name_of(self, value):
        """Gets the name of `value`.
        
        Searches `self.__dict__` for a reference to `value` and returns it.

        Returns:
            (str): A name by which `value` can be accessed from `self`. If
            `self` is accessible by some variable `p`, then
            `eval("p" + p.name_of(value))` must return `value`, assuming
            `p.name_of()` doesn't raise.

        Raises:
            BadParentError: If `value` is not a child of `self`.

        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> p.child = q

            If `q` is a child of `p`, we return the name of `q` in the
            `p`'s `.__dict__`.
            >>> print(p.name_of(q))
            .child

            We are contractually obliged to return a string that when appended
            to `'p'` evaluates to `q`.
            >>> eval('p' + p.name_of(q)) is q
            True
            
        """
        matches = tuple(k for k, v in self.__dict__.items() if v is value)
        if len(matches) == 0:
            raise BadParentError('{value} is not a child of {self}'
                    .format(value=value, self=self))
        assert len(matches) == 1  # no duplicate children!
        return '.' + matches[0]

    def __setattr__(self, name, value):
        """Sets the parent of any new `Tree` attributes appropriately.

        Raises:
            DuplicateNodeError: If `value` is `Tree` and
                already appears in the tree. This forces a spanning 
                tree with no cycles.

        Examples:
            Sets the parent of new children appropriately on assignment.
            >>> p = Tree()
            >>> q = Tree()
            >>> p.child = q
            >>> q.parent is p
            True

            Overwriting a name unsets the parent of the previous occupant.
            >>> r = Tree()
            >>> p.child = r
            >>> q.parent is None
            True

            Attempting to add a `Tree` to the graph twice causes it to be
            surreptitiously shallow-copied:
            >>> r.some_attribute = object()
            >>> p.other_child = r
            >>> p.other_child is r
            False
            >>> p.other_child.some_attribute is r.some_attribute
            True

            However, repeated assignment does not shallow-copy:
            >>> s = Tree()
            >>> p.s = s
            >>> p.s = s
            >>> p.s is s
            True

            Attempting to add a `Tree` to two different graphs causes it
            to be shallow-copied to the new graph.
            >>> s = Tree()
            >>> s.child = r
            >>> p.child is r
            True
            >>> s.child is r
            False
            >>> s.child.some_attribute is r.some_attribute
            True
        
        """
        if name not in ("_Tree__parent", "parent"):
            if isinstance(value, Tree):      
                prev = (getattr(self, name) if hasattr(self, name) else None)
                if value is not prev and value in self.highest_parent:
                    raise DuplicateNodeError('Cannot set \
{long_name}.{name} to {value} because it is already in the tree.'
                            .format(long_name=self.long_name
                                   ,name=name
                                   ,value=value
                                   ))
            self.__maybe_unlink_child(name)  # unlink any existing child

        super().__setattr__(name, value)  # actually set the attribute

        if name not in ("_Tree__parent", "parent"):
            if isinstance(value, Tree):
                value.parent = self

    def __delattr__(self, name):
        """Sets the parent of any removed `Tree` attributes appropriately.
        
        Examples:
            >>> p = Tree()
            >>> q = Tree()
            >>> p.child = q
            >>> q.parent is p
            True
            >>> del p.child
            >>> q.parent is None
            True
        
        """
        if name != '_Tree__parent':  # this check might not be necessary
            self.__maybe_unlink_child(name)
        super().__delattr__(name)
        
    def __maybe_unlink_child(self, name):
        if hasattr(self, name):
            value = getattr(self, name)
            if isinstance(value, Tree):
                value._on_new_parent(None)

    def __iter__(self):
        return BreadthFirstTreeIterator(self)

    def __copy__(self):
        """Shallowly copies self, deeply copying any Tree children.
        
        Examples:
            >>> from copy import copy
            >>> t = Tree()
            >>> t.a = object()
            >>> t.child = Tree()

            Shallow copy returns an entirely new tree, but non-tree
            attributes are shallowly copied (as references).
            >>> copy(t) is t
            False
            >>> copy(t).child is t.child
            False
            >>> copy(t).a is t.a
            True
            
            This allows us to duplicate subtrees:
            >>> t.child2 = copy(t.child)  # no error
          
            This behaviour is inherited correctly:
            >>> class Example(Tree):
            ...     pass
            >>> e = Example()
            >>> e.child = Example()
            >>> e.a = object()
            >>> isinstance(copy(e), Example)
            True
            >>> isinstance(copy(e).child, Example)
            True
            >>> copy(e) is e
            False
            >>> copy(e).child is e.child
            False
            >>> copy(e).a is e.a
            True

        """
        copy = object.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        for k, v in copy.__dict__.items():
            if isinstance(v, Tree) and k != '_Tree__parent':
                copy.__dict__[k] = v.__copy__()
        return copy


class BreadthFirstTreeIterator(object):
    """Walks the tree, breadth first."""
    def __init__(self, tree):
        super().__init__()
        self.queue = [tree]

    def __next__(self):
        if self.queue:
            tree = self.queue.pop(0)
            self.queue.extend(tree.children)
            return tree
        else:
            raise StopIteration
        
    def __iter__(self):
        return self
    
    def __repr__(self):
        return "<{module}.{classname} object {long_name}, id {memory}>"\
                .format(module=__name__, classname=self.__class__,
                        long_name=self.long_name, memory=hex(id(self)))


class TreeWithCache(Tree):
    """A `Tree` with caches at each node. Any node can empty every cache.
    
    Attributes:
        cache (dict): A cache to put things in.
        
    """
    def __init__(self):
        super().__init__()
        self.cache = {}

    def clear_cache(self):
        """Clears this node's cache.
        
        Examples:
            >>> t = TreeWithCache()
            >>> t.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.cache[0] = 123
            >>> t.cache[0]
            123
            >>> t.clear_cache()
            >>> t.cache[0]
            Traceback (most recent call last):
                ..
            KeyError: message
            
        """
        self.cache.clear()

    def clear_tree_caches(self):
        """Clears every cache in the tree.
        
        Examples:
            >>> t = TreeWithCache()
            >>> t.child = TreeWithCache()
            >>> def fill_cache():
            ...     t.cache[0] = 123
            ...     t.child.cache[0] = 456

            Clearing a parent's cache clears its child's cache:
            >>> fill_cache()
            >>> t.clear_tree_caches()
            >>> t.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.child.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
        
            Clearing a child's cache clears its parent's cache:
            >>> fill_cache()
            >>> t.child.clear_tree_caches()
            >>> t.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.child.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message

        """
        for node in self.highest_parent:
            node.clear_cache()

    def clear_subtree_caches(self):
        """Clears the cache of this node and every node lower than it.
        
        Examples:
            >>> t = TreeWithCache()
            >>> t.child = TreeWithCache()
            >>> def fill_cache():
            ...     t.cache[0] = 123
            ...     t.child.cache[0] = 456

            Clearing a parent's cache clears its child's cache:
            >>> fill_cache()
            >>> t.clear_subtree_caches()
            >>> t.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.child.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
        
            Clearing a child's cache does not clear its parent's cache:
            >>> fill_cache()
            >>> t.child.clear_subtree_caches()
            >>> t.cache[0]
            123
            >>> t.child.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message

        """
        for node in self:
            node.clear_cache()

    def clear_ancestor_caches(self):
        """Clears the caches of direct ancestors.
        
        Examples:
            >>> t = TreeWithCache()
            >>> t.child_0 = TreeWithCache()
            >>> t.child_1 = TreeWithCache()
            >>> t.child_0.child = TreeWithCache()
            >>> def fill_cache():
            ...     t.cache[0] = '123'
            ...     t.child_0.cache[0] = '456'
            ...     t.child_1.cache[0] = '789'
            ...     t.child_0.child.cache[0] = 'ABC'

            Clearing `t.child_0.child`'s ancestor's cache clears
            `t`'s cache and `t.child_0`'s cache, but not `t.child_1`'s or
            `t.child_0.child`'s.
            >>> fill_cache()
            >>> t.child_0.child.clear_ancestor_caches()
            >>> t.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.child_0.cache[0]
            Traceback (most recent call last):
                ...
            KeyError: message
            >>> t.child_1.cache[0]
            '789'
            >>> t.child_0.child.cache[0]
            'ABC'

        """
        if self.parent is not None:
            self.parent.clear_cache()
            self.parent.clear_ancestor_caches()


class Leaf(Tree):
    """A `Tree` that cannot have any children.
    
    Examples:
        A `Leaf` will not automatically add children on attribute assignment:
        >>> l = Leaf()
        >>> l.child = Tree()
        >>> l.children
        ()
        >>> l.child.parent is None
        True

        Leaves still know who their parent is and what their name is:
        >>> t = Tree()
        >>> t.fallback_name = 't'
        >>> t.leaf = Leaf()
        >>> t.leaf.parent is t
        True
        >>> t.leaf.name
        '.leaf'
        >>> t.leaf.long_name
        't.leaf'

    """
    def __init__(self):
        super().__init__()

    @property
    @overrides
    def children(self):
        return ()
    
    @overrides
    def __setattr__(self, name, value):
        """Sidestep `Tree.__setattr__` so parents aren't fiddled with."""
        object.__setattr__(self, name, value)

    @overrides
    def __delattr__(self, name):
        """Sidestep `Tree.__delattr__` so parents aren't fiddled with."""
        object.__delattr__(self, name)

    @overrides
    def __copy__(self):
        """Shallowly copies self.

        Examples:
            >>> from copy import copy
            >>> l = Leaf()
            >>> l.child = Tree()
            >>> copy(l).child is l.child  # attributes are not children
            True

        """
        copy = object.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        return copy


class ListTree(Tree, MutableSequence):
    """A `Tree` that uses numeric indexes to track its children.
    
    Examples:
        `ListTree`s know their parent and their name:
        >>> root = Tree()
        >>> root.fallback_name = "root"
        >>> root.list = ListTree()
        >>> root.list.parent is root
        True
        >>> print(root.list.long_name)
        root.list

        Children can be assigned to the list through normal sequence
        assignment:
        >>> child = Tree()
        >>> root.list.append(child)
        >>> child in root.list.children
        True

        Attempting to add a `Tree` to the graph twice causes it to be
        surreptitiously shallow-copied:
        >>> child.some_attribute = object()
        >>> root.list.append(child)
        >>> p[1] is child
        False
        >>> p[1].some_attribute is child.some_attribute
        True

        Overwriting a child with itself does nothing:
        >>> root.list[0] = child
        >>> root.list[0] is child
        True

        The name of the child of a `ListTree` is its index:
        >>> print(child.name)
        [0]
        >>> print(child.long_name)
        root.list[0]

        Assigning a child to another tree shallow-copies it to the other tree.
        >>> t = Tree()
        >>> t.child = child
        >>> t.child is child
        False
        >>> t.child.some_attribute is child.some_attribute
        True
        >>> child.parent is root.list
        True

        Overwriting a child doesn't cause a dangling parent:
        >>> child_a = Tree()
        >>> child_b = Tree()
        >>> root.list.append(child_a)
        >>> root.list[0] = child_b
        >>> child_a.parent is None
        True

    """
    def __init__(self):
        super().__init__()
        self._children = []

    @property
    @overrides
    def children(self):
        return tuple(self._children)

    @overrides
    def name_of(self, child):
        if child in self._children:
            return "[{:d}]".format(self._children.index(child))
        else:
            raise BadParentError("{} is not a child of {}"
                    .format(child.long_name, self.long_name))

    @overrides
    def _unregister_child(self, child):
        if child in self._children:
            self._children.remove(child)
        else:
            raise BadParentError("{} is not a child of {}"
                    .format(child.long_name, self.long_name))

    @overrides
    def insert(self, index, value):
        self.__check(value)
        self._children.insert(index, value)
        value.parent = self

    @overrides
    def __getitem__(self, key):
        return self._children[key]

    @overrides
    def __setitem__(self, key, value):
        if self._children[key] != value:
            self.__check(value)
            self._children[key]._on_new_parent(None)
            self._children[key] = value
            value.parent = self

    @overrides
    def __delitem__(self, key):
        self._children[key]._on_new_parent(None)
        del self._children[key]

    def __check(self, value):
        if not isinstance(value, Tree):
            raise TypeError("ListTree may only contain Tree objects, not {}"
                    .format(type(value)))
        if value in self.highest_parent:
            raise DuplicateNodeError("Could not add {} as a child of {} \
because it is already present in the tree.".format(value.long_name,
self.long_name))

    @overrides
    def __len__(self):
        return len(self._children)

    @overrides
    def __setattr__(self, name, value):
        """Sidestep `Tree.__setattr__` so parents aren't fiddled with."""
        object.__setattr__(self, name, value)

    @overrides
    def __delattr__(self, name):
        """Sidestep `Tree.__delattr__` so parents aren't fiddled with."""
        object.__delattr__(self, name)

    @overrides
    def __copy__(self):
        """Shallowly copies the object but deeply copies children.

        Examples:
            >>> from copy import copy
            >>> t = ListTree()
            >>> t.a = object()
            >>> t.child = Tree()
            >>> t.append(Tree())

            Shallow copy returns an entirely new tree, but non-tree
            attributes are shallowly copied (as references).
            >>> copy(t) is t
            False
            >>> copy(t)[0] is t[0]
            False
            >>> copy(t).a is t.a
            True
            >>> copy(t).child is t.child  # attributes are not children
            True
            
            This allows us to duplicate subtrees:
            >>> t.append(copy(t[0]))  # no error

            This behaviour is inherited correctly:
            >>> class Example(ListTree):
            ...     pass
            >>> e = Example()
            >>> e.append(Example())
            >>> e.a = object()
            >>> isinstance(copy(e), Example)
            True
            >>> isinstance(copy(e)[0], Example)
            True
            >>> copy(e) is e
            False
            >>> copy(e)[0] is e[0]
            False
            >>> copy(e).a is e.a
            True

        """
        copy = object.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        copy._children = [tree.__copy__() for tree in self._children]
        return copy

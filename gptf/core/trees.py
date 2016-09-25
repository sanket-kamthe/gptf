"""Provides the `Tree` class for creating trees of objects."""
from builtins import super, object
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import MutableSequence, Iterable
from functools import wraps
from overrides import overrides

from .utils import LRUCache

class BadParentError(Exception):
    """Raised when a parent does not have a reference to a requested child."""

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
    
class Tree(with_metaclass(ABCMeta, Iterable)):
    """Base class for trees."""
    def __init__(self):
        """Creates a new parentable with no parent."""
        super().__init__()
        self.fallback_name = "unnamed"
        self.__parent = None

    @abstractproperty
    def children(self):
        """Returns the children of this object at the time of calling.

        Returns:
            (tuple) The current children of the object.

        """
        NotImplemented

    @abstractmethod
    def copy(self):
        """Shallow-copy `self`, shallow-copying any `Tree` children."""
        NotImplemented

    def __copy__(self):
        """See `.copy()`. Renamed here so that copy.copy can find it."""
        return self.copy()

    @abstractmethod
    def _unregister_child(self, child):
        """Removes any links between `self` and `child`.
        
        Raises:
            BadParentError: if `child` is not a child of this `Tree`.

        """
        pass

    @property
    def parent(self):
        """Returns the direct parent."""
        return self._get_parent()

    @parent.setter
    def parent(self, value):
        """Sets the parent of a `Tree`.

        If we already had a parent, unlinks `self` from its previous parent
        before linking it to the new one.

        NB: It is _highly unlikely_ that you will need to set this property
        yourself. Setting this property, unless overridden in some parent 
        class, does not register `self` as a child of `value`, which could
        cause unexpected behaviour.

        """
        if self.parent is not None:
            self.parent._unregister_child(self)
        self._set_parent(value)

    def _get_parent(self):
        """Gets the object's direct parent."""
        return self.__parent

    def _set_parent(self, value):
        """Sets the object's direct parent.
        
        NB: There is no guarantee that when this method is called
        `self in self.parent.children` will be `True`. Accessing
        `self.name` or `self.long_name` is therefore unsafe inside
        this method.
        
        """
        self.__parent = value

    @property
    def highest_parent(self):
        """Returns the highest parent in the tree.

        Examples:
            >>> class Example(Tree):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._children = []
            ...     def add_child(self, child):
            ...         self._children.append(child)
            ...         child.parent = self
            ...     @property
            ...     def children(self):
            ...         return tuple(self._children)
            ...     def copy(self):
            ...         copy = self.__new__(type(self))
            ...         copy.__dict__ = self.__dict__.copy()
            ...         copy._children = [c.copy() for c in self._children]
            ...         copy._set_parent(None)
            ...         return copy
            ...     def _unregister_child(self, child):
            ...         if child in self._children:
            ...             self._children.remove(child)
            ...             child._set_parent(None)
            ...         else:
            ...             raise BadParentError

            An object at the top of the tree is its own highest parent.
            >>> p = Example()
            >>> p.highest_parent is p
            True

            Objects lower down in the tree recurse up the tree to find their
            highest parent.
            >>> p.add_child(Example())
            >>> p.children[0].add_child(Example())
            >>> p.children[0].children[0].highest_parent is p
            True

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
            >>> class Example(Tree):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._children = []
            ...     def add_child(self, child):
            ...         self._children.append(child)
            ...         child.parent = self
            ...     @property
            ...     def children(self):
            ...         return tuple(self._children)
            ...     def copy(self):
            ...         copy = self.__new__(type(self))
            ...         copy.__dict__ = self.__dict__.copy()
            ...         copy._children = [c.copy() for c in self._children]
            ...         copy._set_parent(None)
            ...         return copy
            ...     def _unregister_child(self, child):
            ...         if child in self._children:
            ...             self._children.remove(child)
            ...             child._set_parent(None)
            ...         else:
            ...             raise BadParentError
            >>> p = Example()
            >>> q = Example()
            >>> p.add_child(Example())
            >>> p.children[0].add_child(q)
            >>> print(q.long_name)  # we are not in __main__
            unnamed.children[0].children[0]
        
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
            >>> class Example(Tree):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._children = []
            ...     def add_child(self, child):
            ...         self._children.append(child)
            ...         child.parent = self
            ...     @property
            ...     def children(self):
            ...         return tuple(self._children)
            ...     def copy(self):
            ...         copy = self.__new__(type(self))
            ...         copy.__dict__ = self.__dict__.copy()
            ...         copy._children = [c.copy() for c in self._children]
            ...         copy._set_parent(None)
            ...         return copy
            ...     def _unregister_child(self, child):
            ...         if child in self._children:
            ...             self._children.remove(child)
            ...             child._set_parent(None)
            ...         else:
            ...             raise BadParentError

            Objects at the top of the tree return their name in the global
            namespace, if only one such name exists.
            >>> p = Example()
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
            >>> q = Example()
            >>> p.add_child(q)
            >>> print(q.name)
            .children[0]
            
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

        Returns:
            (str): A name by which `value` can be accessed from `self`. If
            `self` is accessible by some variable `p`, then
            `eval("p" + p.name_of(value))` must return `value`, assuming
            `p.name_of()` doesn't raise.

        Raises:
            BadParentError: If `value` is not a child of `self`.

        Examples:
            >>> class Example(Tree):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._children = []
            ...     def add_child(self, child):
            ...         self._children.append(child)
            ...         child.parent = self
            ...     @property
            ...     def children(self):
            ...         return tuple(self._children)
            ...     def copy(self):
            ...         copy = self.__new__(type(self))
            ...         copy.__dict__ = self.__dict__.copy()
            ...         copy._children = [c.copy() for c in self._children]
            ...         copy._set_parent(None)
            ...         return copy
            ...     def _unregister_child(self, child):
            ...         if child in self._children:
            ...             self._children.remove(child)
            ...             child._set_parent(None)
            ...         else:
            ...             raise BadParentError

            In the default implementation, this returns `'.children[x]'`
            where `x` is `self.children.index(value)`.
            >>> p = Example()
            >>> q = Example()
            >>> p.add_child(q)
            >>> p.name_of(q)
            '.children[0]'

        """
        children = self.children
        if value in children:
            return '.children[{}]'.format(self.children.index(value))
        else:
            raise BadParentError('{} is not a child of {}'
                    .format(value, self))
        
    def __iter__(self):
        return BreadthFirstTreeIterator(self)

    def __repr__(self):
        return "<{module}.{classname} object, fallback_name {long_name}, \
id {memory}>".format(module=self.__module__, classname=self.__class__.__name__,
long_name=self.fallback_name, memory=hex(id(self)))


class Leaf(Tree):
    """A `Tree` that cannot have any children.
    
    Examples:
        A `Leaf` will not automatically add children on attribute assignment:
        >>> l = Leaf()
        >>> l.child = AttributeTree()
        >>> l.children
        ()
        >>> l.child.parent is None
        True

        Leaves still know who their parent is and what their name is:
        >>> t = AttributeTree()
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
    def copy(self):
        """Shallowly copies self.

        Examples:
            >>> l = Leaf()
            >>> l.child = AttributeTree()
            >>> l.copy().child is l.child  # attributes are not children
            True

        """
        copy = self.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        super(Leaf, copy)._set_parent(None)
        return copy

    @overrides
    def _unregister_child(self, child):
        raise BadParentError('Leaf objects cannot have children.')

class AttributeTree(Tree):
    """A tree that uses attributes to assign/track children.
    
    A tree can be constructed by simply assigning `Tree`s to the
    attributes of an `AttributeTree`:
        
        >>> p = AttributeTree()
        >>> # doctest does not run these tests in __main__, so we give p
        >>> # a sensible fallback name
        >>> p.fallback_name = 'root'
        >>> p.child_a = AttributeTree()
        >>> p.child_b = AttributeTree()
        >>> p.child_b.leaf = AttributeTree()

    The `long_name()` method may then be used to get the full path of an 
    object within the tree.

        >>> print(p.child_a.long_name)
        root.child_a
        >>> print(p.child_b.leaf.long_name)
        root.child_b.leaf
    
    Objects in the tree must be unique. No object may be its own ancestor
    or sibling. Attempts to add a `Tree` attribute that would result in 
    such a graph instead result in that `Tree` being shallow-copied.

        >>> p = AttributeTree()
        >>> p.attribute = object()
        >>> p.child = p
        >>> p.child is p
        False
        >>> p.child.attribute is p.attribute  # true for non-Tree attributes
        True

        >>> q = AttributeTree()
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
    @overrides
    def _unregister_child(self, child):
        """Removes any links between `self` and `child`.
        
        Raises:
            BadParentError: if `child` is not a child of this `AttributeTree`.

        """
        matches = tuple(k for k, v in self.__dict__.items() if v is child)
        if len(matches) == 0:
            raise BadParentError('{value} is not a child of {self}'
                    .format(value=child, self=self))
        assert len(matches) == 1  # no duplicate children!
        delattr(self, matches[0])

    @property
    @overrides
    def children(self):
        """Returns the children of this object at the time of calling.

        Returns:
            (tuple) The current children of the object.

        Examples:
            >>> p = AttributeTree()
            >>> q = AttributeTree()
            >>> r = AttributeTree()
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

    @overrides
    def copy(self):
        """Shallowly copies self, deeply copying any Tree children.
        
        Examples:
            >>> t = AttributeTree()
            >>> t.a = object()
            >>> t.child = AttributeTree()

            Shallow copy returns an entirely new tree, but non-tree
            attributes are shallowly copied (as references).
            >>> copy = t.copy()
            >>> copy is t
            False
            >>> copy.child is t.child
            False
            >>> copy.child.parent is copy
            True
            >>> copy.a is t.a
            True

            The parent of the new copy is `None`.
            >>> t.child.copy().parent is None
            True
          
            This behaviour is inherited correctly:
            >>> class Example(AttributeTree):
            ...     pass
            >>> e = Example()
            >>> e.child = Example()
            >>> e.a = object()
            >>> copy = e.copy()
            >>> isinstance(copy, Example)
            True
            >>> isinstance(copy.child, Example)
            True
            >>> copy is e
            False
            >>> copy.child is e.child
            False
            >>> copy.a is e.a
            True

        """
        copy = self.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        tree_children = []
        for k, v in self.__dict__.items():
            if isinstance(v, Tree) and k != '_Tree__parent':
                del copy.__dict__[k]
                tree_children.append((k, v))
        super(AttributeTree, copy)._set_parent(None)
        for k, v in tree_children:
            setattr(copy, k, v.copy())
        return copy

    @overrides
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
            >>> p = AttributeTree()
            >>> q = AttributeTree()
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

        Examples:
            Sets the parent of new children appropriately on assignment.
            >>> p = AttributeTree()
            >>> q = AttributeTree()
            >>> p.child = q
            >>> q.parent is p
            True

            Overwriting a name unsets the parent of the previous occupant.
            >>> r = AttributeTree()
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
            >>> s = AttributeTree()
            >>> p.s = s
            >>> p.s = s
            >>> p.s is s
            True

            Attempting to add a `Tree` to two different graphs causes it
            to be shallow-copied to the new graph.
            >>> s = AttributeTree()
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
                if (value is not prev and (value.parent is not None 
                    or value is self.highest_parent)):
                    value = value.copy()
            self.__maybe_unlink_child(name)  # unlink any existing child

        super().__setattr__(name, value)  # actually set the attribute

        if name not in ("_Tree__parent", "parent"):
            if isinstance(value, Tree):
                value.parent = self

    def __delattr__(self, name):
        """Sets the parent of any removed `Tree` attributes appropriately.
        
        Examples:
            >>> p = AttributeTree()
            >>> q = AttributeTree()
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
                value._set_parent(None)

class ListTree(Tree, MutableSequence):
    """A `Tree` that uses numeric indexes to track its children.
    
    Examples:
        `ListTree`s know their parent and their name:
        >>> root = AttributeTree()
        >>> root.fallback_name = "root"
        >>> root.list = ListTree()
        >>> root.list.parent is root
        True
        >>> print(root.list.long_name)
        root.list

        Children can be assigned to the list through normal sequence
        assignment:
        >>> child = AttributeTree()
        >>> root.list.append(child)
        >>> child in root.list.children
        True
        >>> len(root.list)
        1

        Attempting to add a `Tree` to the graph twice causes it to be
        surreptitiously shallow-copied:
        >>> child.some_attribute = object()
        >>> root.list.append(child)
        >>> len(root.list)
        2
        >>> root.list[1] is child
        False
        >>> root.list[1].some_attribute is child.some_attribute
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
        >>> t = AttributeTree()
        >>> t.child = child
        >>> t.child is child
        False
        >>> t.child.some_attribute is child.some_attribute
        True
        >>> child.parent is root.list
        True

        Overwriting a child doesn't cause a dangling parent:
        >>> root.list = ListTree()
        >>> child_a = AttributeTree()
        >>> child_b = AttributeTree()
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
        value = self.__check(value)
        self._children.insert(index, value)
        value.parent = self

    @overrides
    def __getitem__(self, key):
        return self._children[key]

    @overrides
    def __setitem__(self, key, value):
        if self._children[key] is not value:
            value = self.__check(value)
            self._children[key]._set_parent(None)
            self._children[key] = value
            value.parent = self

    @overrides
    def __delitem__(self, key):
        self._children[key]._set_parent(None)
        del self._children[key]

    def __check(self, value):
        if not isinstance(value, Tree):
            raise TypeError("ListTree may only contain Tree objects, not {}"
                    .format(type(value)))
        if value is self.highest_parent or value.parent is not None:
            return value.copy()
        return value

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
    def copy(self):
        """Shallowly copies the object but deeply copies children.

        Examples:
            >>> t = ListTree()
            >>> t.a = object()
            >>> t.child = AttributeTree()
            >>> t.append(AttributeTree())

            Shallow copy returns an entirely new tree, but non-tree
            attributes are shallowly copied (as references).
            >>> copy = t.copy()
            >>> copy is t
            False
            >>> copy[0] is t[0]
            False
            >>> copy[0].parent is copy
            True
            >>> copy.a is t.a
            True
            >>> copy.child is t.child  # attributes are not children
            True
            
            This behaviour is inherited correctly:
            >>> class Example(ListTree):
            ...     pass
            >>> e = Example()
            >>> e.append(Example())
            >>> e.a = object()
            >>> copy = e.copy()
            >>> isinstance(copy, Example)
            True
            >>> isinstance(copy[0], Example)
            True
            >>> copy is e
            False
            >>> copy[0] is e[0]
            False
            >>> copy.a is e.a
            True

        """
        copy = self.__new__(type(self))
        copy.__dict__ = self.__dict__.copy()
        copy._children = []
        super(ListTree, copy)._set_parent(None)
        for child in self.children:
            copy.append(child.copy())
        return copy

class TreeWithCache(Tree):
    """A `Tree` with caches at each node. Any node can empty every cache.
    
    Attributes:
        cache (dict): A cache to put things in.
        
    """
    def __init__(self):
        super().__init__()
        self.cache = {}

    @overrides
    def copy(self):
        """Copies have empty caches."""
        dupe = super().copy()
        dupe.cache = {}
        return dupe

    def clear_cache(self):
        """Clears this node's cache.
        
        Examples:
            >>> class Example(AttributeTree, TreeWithCache):
            ...     pass
            >>> t = Example()
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
            >>> class Example(AttributeTree, TreeWithCache):
            ...     pass
            >>> t = Example()
            >>> t.child = Example()
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
            >>> class Example(AttributeTree, TreeWithCache):
            ...     pass
            >>> t = Example()
            >>> t.child = Example()
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
            >>> class Example(AttributeTree, TreeWithCache):
            ...     pass
            >>> t = Example()
            >>> t.child_0 = Example()
            >>> t.child_1 = Example()
            >>> t.child_0.child = Example()
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

def cache_method(capacity=128):
    """Adds caching to a method of a `TreeWithCache`.

    Stores an `LRUCache` which maps from method call arguments to
    return values in the instance's cache. This keeps the caches
    of each instance seperate, and allows us to clear them using
    `instance.cache_clear()`.

    Note: if the arguments are not hashable, we skip caching and
    just return the value of the method.

    Args:
        capacity (int): The capacity of the cache. If the size of
            the cache exceeds the capacity, the least recently used
            (stored / retreived) arguments will be evicted.

    """
    def decorator(method):
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            cache_name = '__tf_method_cache_{}'.format(id(method))
            if cache_name not in instance.cache:
                instance.cache[cache_name] = LRUCache(capacity)
            method_cache = instance.cache[cache_name]
            key = (args, frozenset(kwargs.items()))
            try:
                hash(key)
            except TypeError:
                return method(instance, *args, **kwargs)
            else:
                if key not in method_cache:
                    method_cache[key] =  method(instance, *args, **kwargs)
                return method_cache[key]
        return wrapper
    return decorator

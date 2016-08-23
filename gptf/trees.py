"""Provides the `Tree` class for creating trees of objects."""
from builtins import super, object
try:  # in case of rogue Python 2.7, use contextlib2 instead of contextlib
    from contextlib import suppress
except ImportError:
    from contextlib2 import suppress

from overrides import overrides

class DuplicateNodeError(Exception):
   """Raised when an attempt is made to construct a bad tree.
    
    A bad `Tree` graph is one where some `Tree` is its own
    ancestor or sibling; that is, some `Tree` appears more than
    once in the graph.
    
    """

class BadParentError(Exception):
    """Raised when a parent does not have a reference to a requested child."""


#TODO: Let trees have multiple parents?
class Tree(object):
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
    or sibling. Attempting to construct such a graph results in a
    `DuplicateNodeError`.

        >>> p = Tree()
        >>> p.child = p
        Traceback (most recent call last):
            ...
        gptf.trees.DuplicateNodeError: message

        >>> q = Tree()
        >>> p.child_a = q
        >>> p.child_b = q
        Traceback (most recent call last):
            ...
        gptf.trees.DuplicateNodeError: message

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
            self.parent.__unregister_child(self)
        self.on_new_parent(value)

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
        with suppress(KeyError):
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

    def __unregister_child(self, child):
        """Removes any links between `self` and `child`.
        
        Raises:
            BadParentError: if `child` is not a child of this `Tree`.

        """
        matches = tuple(k for k, v in self.__dict__.items() if v is child)
        if len(matches) == 0:
            raise BadParentError('{value} is not a child of {self}'
                    .format(value=child, self=self))
        assert len(matches) == 1  # no duplicate children!
        self.__delattr__(matches[0])

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

            Attempting to add a `Tree` to the graph twice causes an error:
            >>> p.other_child = r
            Traceback (most recent call last):
                ...
            gptf.trees.DuplicateNodeError: message

            Attempting to add a `Tree` to two different graphs causes it
            to be removed from the first one.
            >>> s = Tree()
            >>> s.child = r
            >>> r in p.children
            False
            >>> p.child
            Traceback (most recent call last):
                ...
            AttributeError: message
            >>> p.child = s.child
            >>> r in s.children
            False
            >>> r.child
            Traceback (most recent call last):
                ...
            AttributeError: message
        
        """
        if name not in ("_Tree__parent", "parent"):
            self.__maybe_unlink_child(name)  # unlink any existing child

            if isinstance(value, Tree):      
                if value in self.highest_parent:
                    raise DuplicateNodeError('Cannot set \
{long_name}.{name} to {value} because it is already in the tree.'
                            .format(long_name=self.long_name
                                   ,name=name
                                   ,value=value
                                   ))

        super().__setattr__(name, value)  # actually set the attribute

        if name not in ("_Tree__parent", "parent"):
            if isinstance(value, Tree):
                value.parent = self
                #if value.parent is not None:
                #    value.parent.__unregister_child(value)
                #value.on_new_parent(self)  # set value.parent accordingly

    def on_new_parent(self, new_parent):
        """Called when an object's direct parent is set.
        
        Sets `self.parent` to `new_parent`.
        
        """
        self.__parent = new_parent

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
        with suppress(AttributeError):
            value = self.__getattribute__(name)
            if isinstance(value, Tree):
                value.on_new_parent(None)

    def __iter__(self):
        return BreadthFirstTreeIterator(self)


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
        """Sidestep `Tree.__setattr__` so no children are added."""
        object.__setattr__(self, name, value)

    @overrides
    def __delattr__(self, name):
        """Sidestep `Tree.__delattr__` so no children are added."""
        object.__delattr__(self, name)

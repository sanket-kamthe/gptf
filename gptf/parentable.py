"""Provides the `Parentable` class for creating trees of objects."""
from builtins import super
try:  # in case of rogue Python 2.7, use contextlib2 instead of contextlib
    from contextlib import suppress
except ImportError:
    from contextlib2 import suppress

class DuplicateParentableError(Exception):
   """Raised when an attempt is made to construct a bad `Parentable` graph.
    
    A bad `Parentable` graph is one where some `Parentable` is its own
    ancestor or sibling; that is, some `Parentable` appears more than
    once in the graph.
    
    """

class BadParentError(Exception):
    """Raised when a parent does not have a reference to a requested child."""

class Parentable(object):
    """Provides methods for getting the full path to an object in a tree.
    
    A tree can be constructed by simply assigning `Parentable`s to each other's
    attributes:
        
        >>> p = Parentable()
        >>> p.fallback_name = 'root'  # give p a name
        >>> p.child_a = Parentable()
        >>> p.child_b = Parentable()
        >>> p.child_b.leaf = Parentable()

    The `long_name()` method may then be used to get a `Parentable`'s full
    path within the tree.

        >>> p.child_a.long_name
        'root.child_a'
        >>> p.child_b.leaf.long_name
        'root.child_b.leaf'
    
    Objects in the tree must be unique. No object may be its own ancestor
    or sibling. Attempting to construct such a graph results in a
    `DuplicateParentableError`.

        >>> p = Parentable()
        >>> p.child = p
        Traceback (most recent call last):
            ...
        DuplicateParentableError: ...

        >>> q = Parentable()
        >>> p.child_a = q
        >>> p.child_b = q
        Traceback (most recent call last):
            ...
        DuplicateParentableError: ...

    Attributes:
        fallback_name (str): The name that this object has when it is the
            highest object in the tree and the `name()` method has not been
            overloaded. Defaults to "unnamed".
    
    """

    def __init__(self):
        """Creates a new parentable with no parent."""
        self.fallback_name = "unnamed"
        self._parent = None

    @property
    def parent(self):
        """Returns the direct parent.

        Examples:
            >>> p = Parentable()
            >>> q = Parentable()
            >>> p.child = q
            >>> q.parent is p
            True

        """
        return self._parent

    @property
    def highest_parent(self):
        """Returns the highest parent in the tree.

        Examples:
            An object at the top of the tree is its own highest parent.
            >>> p = Parentable()
            >>> p.highest_parent is p
            True

            Objects lower down in the tree recurse up the tree to find their
            highest parent.
            >>> p.q = Parentable()
            >>> p.q.r = Parentable()
            >>> p.q.r.highest_parent is p
            True

            see that function's
            documentation for how argumewnt
        """
        if self._parent is None:
            return self
        else:
            return self._parent.highest_parent

    @property
    def long_name(self):
        """Gets the long name of `self`.

        Returns:
            (str): The full path to the object in the `Parentable` tree.
        
        Examples:
            >>> p = Parentable()
            >>> q = Parentable()
            >>> p.child = Parentable()
            >>> p.child.leaf = q
            >>> q.long_name
            'unnamed.child.leaf'
        
        """
        if self._parent is None:
            return self.name
        else:
            return self._parent.long_name + self.name

    @property
    def name(self):
        """Gets the name by which this object's parent refers to it.
        
        If there is no parent, returns `self.fallback_name`.
        
        Examples:
            Objects at the top of the tree return their fallback name if
            `name()` is not overloaded.
            >>> p = Parentable()
            >>> p.name == p.fallback_name
            True

            Otherwise, an object's name is its name in its parent's __dict__,
            if said parent's `.name_of()` has not been overloaded.
            >>> q = Parentable()
            >>> p.child = q
            >>> q.name
            '.child'
            
        """
        if self._parent is None:
            return self.fallback_name
        else:
            return self._parent.name_of(self)

    def name_of(self, value):
        """Gets the name of `value`.

        Returns:
            (str): A name by which `value` can be accessed from `self`. If
            `self` is accessible by some variable `p`, then
            `eval("p." + p.name_of(value))` must return `value`, assuming
            `p.name_of()` doesn't raise.

        Examples:
            >>> p = Parentable()
            >>> q = Parentable()
            >>> p.child = q

            If `q` is a child of `p`, we return the name of `q` in the
            `p`'s `.__dict__`.
            >>> p.name_of(q)
            '.child'

            We are contractually obliged to return a string that when appended
            to `'p'` evaluates to `q`.
            >>> eval('p' + p.name_of(q)) is q
            True
            
        Raises:
            BadParentError: If `value` is not a child of `self`.

        """
        matches = tuple(k for k, v in self.__dict__.items() if v is value)
        if len(matches) == 0:
            raise BadParentError('{value} is not a child of {self}'
                    .format(value=value, self=self))
        assert len(matches) == 1  # no duplicate children!
        return '.' + matches[0]

    def _in_subtree(self, value):
        """Checks if `value` is in the tree formed when `self` is the root.
    
        Args:
            value (Parentable): the value to check for.

        Returns:
            (bool): True iff `value` is present in the subtree.

        """
        if self is value:
            return True

        children = dict(self.__dict__)
        del children['_parent']
        for p in children.values():
            if isinstance(p, Parentable) and p._in_subtree(value):
                return True

        return False
        
    def __setattr__(self, name, value):
        """Sets the parent of any new Parentable attributes appropriately.

        Raises:
            DuplicateParentableError: If `value` is `Parentable` and
                already appears in the tree. This forces a spanning 
                tree with no cycles.

        Examples:
            Sets the parent of new children appropriately on assignment.
            >>> p = Parentable()
            >>> q = Parentable()
            >>> p.child = q
            >>> q.parent is p
            True

            Overwriting a name unsets the parent of the previous occupant.
            >>> r = Parentable()
            >>> p.child = r
            >>> q.parent is None
            True

            Attempting to add a `Parentable` to the graph twice causes an error:
            >>> p.other_child = r
            Traceback (most recent call last):
                ...
            DuplicateParentableError: ...

        
        """
        # set the parent of an existing Parentable attribute to `None`
        with suppress(AttributeError):
            previous_value = self.__getattribute__(name)
            if isinstance(previous_value, Parentable):
                previous_value._parent = None

        if name != "_parent" and isinstance(value, Parentable):
            if self.highest_parent._in_subtree(value):
                raise DuplicateParentableError('Cannot set {long_name}.{name} \
to {value} because it is already in the tree.'
                        .format(long_name=self.long_name
                               ,name=name
                               ,value=value
                               ))
            else:
                value._parent = self

        super().__setattr__(name, value)

if __name__ == "__main__":
    import doctest
    flags = doctest.NORMALIZE_WHITESPACE \
            | doctest.ELLIPSIS \
            | doctest.REPORT_NDIFF \
            #| doctest.REPORT_ONLY_FIRST_FAILURE
    failed, total = doctest.testmod(optionflags=flags)
    total_s = '' if total == 1 else 's'
    failed_s = '' if failed == 1 else 's'
    print("{} test{} ran, {} failure{}"
            .format(total, total_s, failed, failed_s))

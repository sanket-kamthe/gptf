"""A collection of hacks for TensorFlow.

Hopefully, as TensorFlow matures as a library, more of these can be removed.

"""
import tensorflow as tf

def eye(shape, dtype=tf.float64):
    """Constructs an eye matrix of the requested shape.
    
    Args:
        shape (int | (int, int)): The shape to construct. If `int`, a square
            matrix with `shape` rows and `shape` columns will be constructed.
            Else, a matrix with `shape[0]` rows and `shape[1]` columns will 
            be constructed.

    Returns:
        (tf.Tensor): A matrix with the requested shape and 1s on the diagonal.

    """
    with tf.name_scope("gptf.tfhacks.eye") as scope:
        try:
            assert len(shape) == 2  # raises TypeError if shape is int
            n = tf.minimum(shape[0], shape[1])
            identity = tf.diag(tf.ones([n], dtype=dtype))
            padrows = tf.to_int32(shape[0]) - n
            padcolumns = tf.to_int32(shape[1]) - n
            paddings = [[0, padrows], [0, padcolumns]]
            return tf.pad(identity, paddings, name=scope)
        except TypeError:
            # shape is int
            return tf.diag(tf.ones([shape], dtype=dtype), name=scope)

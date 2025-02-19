import jax.numpy as jnp
import jax.tree_util


def tree_slice(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def tree_concatenate(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *trees)


def tree_unsqueeze(tree):
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), tree)


def tree_split_first_dim(tree, leading):
    def fn(x):
        old_shape = x.shape
        new_shape = (leading, int(old_shape[0] / leading), *old_shape[1:])
        return x.reshape(*new_shape)

    return jax.tree_util.tree_map(fn, tree)


def tree_merge_first_dim(tree):
    def fn(x):
        old_shape = x.shape
        new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
        return x.reshape(*new_shape)

    return jax.tree_util.tree_map(fn, tree)

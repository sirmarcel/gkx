import jax.numpy as jnp
import jax.tree_util


def tree_slice(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def tree_concatenate(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *trees)

def tree_unsqueeze(tree):
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), tree)
"""Importing this package registers every task bucket via @tasks.register.

Add a new task = add a bucket dir with an __init__.py that registers itself,
then import it here.
"""

from . import newsletter, podcast, youtube  # noqa: F401

"""Fixtures shared by the torch test modules: the synthetic checkpoints of test_forward.CONFIGS,
written once per session, and stored, the verified pinned checkpoints (test_weights)."""

import pytest
from test_forward import CONFIGS, random_tensors
from test_weights import stored, write_checkpoint  # noqa: F401 - stored is a fixture


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory):
    directories = {}
    for index, (name, doc) in enumerate(CONFIGS.items()):
        directory = tmp_path_factory.mktemp(name)
        write_checkpoint(directory, doc, random_tensors(doc, index))
        directories[name] = directory
    return directories

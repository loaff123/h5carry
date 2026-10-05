"""Original benign fixtures; call constructors only inside a bounded worker."""
from examples.three_run_archive import create_demo, create_family, create_plain_control

# A short compatibility name for tests that read the frozen specification.
fixture = create_family

__all__ = ['create_demo', 'create_family', 'create_plain_control', 'fixture']

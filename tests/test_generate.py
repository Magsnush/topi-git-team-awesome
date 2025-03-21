"""Unit tests for data generation."""

import sys

import numpy as np
import pytest

# ↓ when you have a proper package in exercise C, replace this hack
sys.path.append("src/")
from generate import Hill, SkiJump


class TestHill:
    """Check Hill."""

    def test_y(self):
        """Check slope"""
        # this is an already working example:
        # We create a fixed configuration ...
        hill = Hill(offset=-1.0, slope=-1.0)
        # ... and then check it is working as intended
        # Recall that testing float numbers is actually complicated (why?) and for that purpose Numpy is
        # providing a set of handy functions in numpy.testing -
        # see also https://numpy.org/doc/stable/reference/routines.testing.html#module-numpy.testing
        np.testing.assert_allclose(hill.y(0.0), -1.0)
        np.testing.assert_allclose(hill.y(1.0), -2.0)


class TestSkiJump:
    """Check SkiJump."""

    # Pytest offers a built-in method to temporarily deactivate test:
    # https://docs.pytest.org/en/stable/how-to/skipping.html#skipping-test-functions
    # ↓ When you start working on `SkiJump.y` remove the line below ↓ (and similar)
    # @pytest.mark.skip(reason="Not implemented yet.")
    # ↑ When you start working on `SkiJump.y` remove the line above ↑ (and similar)
    def test_y(self):
        """Check trajectory."""
        # Work here in Step 2!
        # Adjust the implementation of SkiJump.y such that these test pass.
        # First a "simple" example
        jump_straight = SkiJump(v0=np.sqrt(0.5), alpha=0.0)
        np.testing.assert_allclose(jump_straight.y(0.0), 0.0)
        np.testing.assert_allclose(jump_straight.y(1.0), -1.0)
        np.testing.assert_allclose(jump_straight.y(2.0), -4.0)
        # Then a "not-so-simple" example
        jump_up = SkiJump(v0=1.0, alpha=np.pi / 4.0)
        np.testing.assert_allclose(jump_up.y(0.5), 0.25)
        np.testing.assert_allclose(jump_up.y(1.0), 0.0)

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_from_json_file(self, tmp_path):
        """Check file config."""
        # Work here in Step 2!
        # Test which involve files are slightly more complex - but of course still doable!
        # The solution in pytest is described here: https://docs.pytest.org/en/stable/how-to/tmp_path.html
        # With the special argument `tmp_path` we get a temporary path provided, which we can then use for our purpose.
        # So, let's first define our target file ...
        file = tmp_path / "test.json"
        # ... then we can fill it with the stuff that we want:
        alpha = -1.0
        v0 = 1.23
        file.write_text(f"""{{"alpha": {alpha},"v0": {v0}}}""", encoding="utf-8")
        # now we can use that for our test:
        jump = SkiJump.from_json_file(file)
        # We can e.g. check that the values are read correctly
        np.testing.assert_allclose(jump.alpha, alpha)
        np.testing.assert_allclose(jump.v0, v0)

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_landing(self):
        """Check landing point."""
        # Work here in Step 2!
        # Design your own test for `SkiJump.landing`!
        # Setup a controlled environment and check against numbers, which you know are correct.
        # If the numbers are non-trivial to obtain, it is a good idea to give a reference.

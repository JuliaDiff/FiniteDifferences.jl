export assert_approx_equal

"""
    assert_approx_equal(x, y, ε_abs, ε_rel[, desc])

Assert that `x` is approximately equal to `y`.

Let `ε_z = ε_abs / ε_rel`. Call `x` and `y` small if `abs(x) + abs(y) < ε_z`, and call `x`
and `y` large otherwise. If this assertion succeeds, then it is guaranteed that
`abs(x - y) < 2ε_rel * (abs(x) + abs(y))` if `x` and `y` are large, and
`abs(x - y) < 2ε_abs` if `x` and `y` are small.

# Arguments
- `x`: First object to compare.
- `y`: Second object to compare.
- `ε_abs`: Absolute tolerance.
- `ε_rel`: Relative tolerance.
- `desc`: Description of the comparison. Omit or set to `false` to have no description.
"""
function assert_approx_equal(x, y, ε_abs, ε_rel, desc)
    if abs(x - y) >= ε_abs + ε_rel * (abs(x) + abs(y))
        msg = "$(desc != false ? "\"$desc\": " : "")large deviation from reference:\n" *
              "  relative error:  $(@sprintf "%.3e" abs(x - y) / (abs(x) + abs(y)))\n" *
              "    tolerance:     $(@sprintf "%.3e" ε_rel)\n" *
              "  absolute error:  $(@sprintf "%.3e" abs(x - y))\n" *
              "    tolerance:     $(@sprintf "%.3e" ε_abs)\n"
        throw(ErrorException(msg))
    end
    return true
end
assert_approx_equal(x, y, ε_abs, ε_rel) = assert_approx_equal(x, y, ε_abs, ε_rel, false)

# Copyright (C) 2025–2026 David Helekal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

using SafeTestsets
using Test

# Keep failures in one group from preventing the remaining groups from running.
# Each SafeTestset isolates imports, helper definitions, and local RNGs.
@testset "BASSGWAS" begin
    @safetestset "Block task execution" begin include("block_tasks_tests.jl") end
    @safetestset "Floating-point state types" begin include("state_type_tests.jl") end
    @safetestset "bvs tests" begin include("wtgs_tests.jl") end
    @safetestset "Exact conditional posteriors" begin include("posterior_tests.jl") end
    @safetestset "Update rates and transitions" begin include("rate_tests.jl") end
    @safetestset "Numerical cache updates" begin include("cache_tests.jl") end
    @safetestset "Continuous conditional draws" begin include("conditional_draw_tests.jl") end
    @safetestset "Dependency caches and workspaces" begin include("workspace_tests.jl") end
    @safetestset "Full-model regressions" begin include("boundary_regression_tests.jl") end
end
